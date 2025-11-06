

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm

from flcore.security.attack.privacy.inference.mia.base import MembershipInferenceAttack
from flcore.security.attack.privacy.inference.mia.utils.models import CNN_MIA, ShadowAttackModel, WhiteBoxAttackModel, ML_CNN, ML_NN, ML_Softmax
from flcore.security.attack.privacy.inference.mia.utils.trainer import shadow

atk_model = {'cnn': ML_CNN,
            'nn': ML_NN,
            'softmax': ML_Softmax, 
            'shadow': ShadowAttackModel}
DATA_TYPE = {
    "resnet18": "vision",
    "logreg": "text",
    "tinybert": "text",
}

class MLleaksAttacker(MembershipInferenceAttack):
    def __init__(self, Server, conf2=None):
        conf_path = "./security/attack/privacy/config/MIA_ML_leaks.yaml"
        super().__init__('ML_leaks', Server, conf_path, conf2, c_model=None)
        batch_size = self.config['train']['batch_size']
        # data_shape = Server.train_data[0][0][0][0].shape
        modality = DATA_TYPE.get(Server.global_model.__class__.__name__.lower(), 'vision')
        if(self.config['type'] == "shadow"):
            if modality == 'vision':
                self.shadow_model = CNN_MIA(input_channel=3, num_classes=self.num_classes)
            else:
                raise NotImplementedError("Shadow model for text data is not implemented yet.")

        self.attack_model = atk_model[self.config['atk_model']](self.num_classes, batch_size)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.attack_model.parameters(), lr=float(self.config['train']['lr']))

    def _get_gradient_size(self, model):
        gradient_size = []
        gradient_list = reversed(list(model.named_parameters()))
        for name, parameter in gradient_list:
            if 'weight' in name:
                gradient_size.append(parameter.shape)

        return gradient_size

    def MIA(self, target):
        aux_len = max(len(self.Server.test_data)-len(target), len(self.Server.test_data)//2)
        aux_data, test_data = torch.utils.data.random_split(self.Server.test_data, [aux_len, len(self.Server.test_data) - aux_len])
        self.preprocess_shadow_data(aux_data, target, test_data)
        self.train_shadow_models()
        self.target_model = self.Server.clients[self.cid].model
        self.train()
        predict, c, y = self.test()
        return predict, c, y

    def _get_data(self, model, inputs, targets, masks = None):
        if masks is None:
            result = self.target_model(inputs)
        else:
            result = self.target_model(inputs, masks).logits
        output, _ = torch.sort(result, descending=True)
        _, predicts = result.max(1)
        prediction = predicts.eq(targets.squeeze()).float()
        return output, prediction

    def train(self):
        self.attack_model.to(self.device)
        self.attack_model.train()
        train_loss = 0
        correct = 0
        total = 0

        final_result = []

        for i in tqdm(range(self.config['train']['epochs'])):
            for batch in self.attack_train_loader:
                if(len(batch) == 3):
                    inputs, targets, members = batch
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    output, prediction = self._get_data(inputs, targets)
                else:
                    inputs, masks, targets, members = batch
                    inputs, masks, targets = inputs.to(self.device), masks.to(self.device), targets.to(self.device)
                    output, prediction = self._get_data(inputs, targets, masks)

                output, prediction = self._get_data(self.shadow_model, inputs, targets)
                output, prediction, members = output.to(self.device), prediction.to(self.device), members.to(self.device)
                results = self.attack_model(output, prediction)
                results = F.softmax(results, dim=1)

                losses = self.criterion(results, members)
                losses.backward()
                self.optimizer.step()

                train_loss += losses.item()
                _, predicted = results.max(1)
                total += members.size(0)
                correct += predicted.eq(members).sum().item()

                final_result.append(1.*correct/total)
            print( 'Train Acc: %.3f%% (%d/%d) | Loss: %.3f' % (100.*correct/total, correct, total, 1.*train_loss/((i+1)*len(self.attack_train_loader))))

        return final_result

    def test(self):
        self.attack_model.to(self.device)
        self.attack_model.eval()
        correct = 0
        total = 0

        final_result = []
        predicts, c, y = [], [], []

        for inputs, targets, members in self.attack_test_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            y.extend(targets.detach().clone().cpu().tolist())
            c.extend(members.detach().clone().cpu().tolist())

            output, label = self._get_data(self.target_model, inputs, targets)
            output, label, members = output.to(self.device), label.to(self.device), members.to(self.device)
            results = self.attack_model(output, label)
          
            with torch.no_grad():
                results = self.attack_model(output, label)
                _, predicted = results.max(1)
                total += members.size(0)
                correct += predicted.eq(members).sum().item()
                
                results = F.softmax(results, dim=1)
                predicts.extend(predicted.detach().clone().cpu().tolist())

            final_result.append(1.*correct/total)
            print( 'Nasr\'s Whitebox Membership Inference Acc: %.3f%% (%d/%d)' % (100.*correct/(1.0*total), correct, total))
        return predicts, c, y

    def train_shadow_models(self, shadow_path = None):
        optimizer = optim.SGD(self.shadow_model.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4)
        criterion = nn.CrossEntropyLoss() 

        s_epoch = self.config['train']['shadow_epochs']
        model = shadow(self.shadow_train_loader, self.shadow_test_loader, self.shadow_model, optimizer, criterion, s_epoch, self.device)
        acc_train = 0
        acc_test = 0
        acc_train = model.train()
        acc_test = model.test()
        
        overfitting = round(acc_train - acc_test, 6)
        print('The overfitting rate is %s' % overfitting)

        if (shadow_path):
            model.saveModel(shadow_path)
        print("saved shadow model!!!")
        print("Finished training!!!")

        return acc_train, acc_test, overfitting
