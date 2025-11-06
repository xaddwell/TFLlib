# whitebox setting in paper:<Comprehensive Privacy Analysis of Deep Learning:
#     Passive and Active White-box Inference Attacks against Centralized 
#     and Federated Learning>
from flcore.security.attack.privacy.inference.mia.base import MembershipInferenceAttack
from flcore.security.attack.privacy.inference.mia.utils.models import WhiteBoxAttackModel

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class WhiteboxPartialAttacker(MembershipInferenceAttack):
    def __init__(self, Server, conf2=None):
        conf_path = "./security/attack/privacy/config/MIA_Nasr.yaml"
        super().__init__("WhiteboxPartialAttacker", Server, conf_path, conf2, c_model=None)
        gradient_size = self._get_gradient_size(Server.global_model)
        total = gradient_size[0][0] // 2 * gradient_size[0][1] // 2

        self.attack_model = WhiteBoxAttackModel(self.num_classes, total)

        self.epochs = self.config['train']['epochs']

        self.optimizer = optim.Adam(self.attack_model.parameters(), lr=1e-3)
        self.target_criterion = nn.CrossEntropyLoss(reduction='none')
        self.attack_criterion = nn.CrossEntropyLoss()       

    def _get_gradient_size(self, model):
        gradient_size = []
        gradient_list = reversed(list(model.named_parameters()))
        for name, parameter in gradient_list:
            if 'weight' in name:
                gradient_size.append(parameter.shape)

        return gradient_size

    def MIA(self, target):
        self.preprocess_train_data(self.Server.test_data, target) 
        self.target_model = self.Server.clients[self.cid].model.to(self.device)
        self.train()
        predict, c, y = self.test()
        return np.array(predict), np.array(c), np.array(y)

    def train(self):
        self.attack_model.to(self.device)
        self.attack_model.train()
        train_loss = 0
        correct = 0
        total = 0

        final_result = []

        for i in tqdm(range(self.epochs)):
            for batch in self.attack_train_loader:
                if(len(batch) == 3):
                    inputs, targets, members = batch
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    output, loss, gradient, label = self._get_data(inputs, targets)
                else:
                    inputs, masks, targets, members = batch
                    inputs, masks, targets = inputs.to(self.device), masks.to(self.device), targets.to(self.device)
                    output, loss, gradient, label = self._get_data(inputs, targets, masks)
                
                output, loss, gradient, label, members = output.to(self.device), loss.to(self.device), gradient.to(self.device), label.to(self.device), members.to(self.device)
                results = self.attack_model(output, loss, gradient, label)
                losses = self.attack_criterion(results, members)
                losses.backward()
                self.optimizer.step()

                train_loss += losses.item()
                _, predicted = results.max(1)
                total += members.size(0)
                correct += predicted.eq(members).sum().item()

                final_result.append(1.*correct/total)
        print( 'Train Acc: %.3f%% (%d/%d) | Loss: %.3f' % (100.*correct/total, correct, total, 1.*train_loss/self.epochs))

        return final_result

    def test(self):
        self.attack_model.eval()
        correct = 0
        total = 0

        final_result = []
        predicts, c, y = [], [], []

        for batch in self.attack_train_loader:
            if(len(batch) == 3):
                inputs, targets, members = batch
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                output, loss, gradient, label = self._get_data(inputs, targets)
            else:
                inputs, masks, targets, members = batch
                inputs, masks, targets = inputs.to(self.device), masks.to(self.device), targets.to(self.device)
                output, loss, gradient, label = self._get_data(inputs, targets, masks)
            output, loss, gradient, label, members = output.to(self.device), loss.to(self.device), gradient.to(self.device), label.to(self.device), members.to(self.device)
            results = self.attack_model(output, loss, gradient, label)
            c.extend(targets.detach().clone().cpu().tolist())
            y.extend(members.detach().clone().cpu().tolist())
            _, predicted = results.max(1)
            predicts.extend(predicted.detach().clone().cpu().tolist())
            total += members.size(0)
            correct += predicted.eq(members).sum().item()

            results = F.softmax(results, dim=1)

            final_result.append(1.*correct/total)
            print( 'Nasr\'s Whitebox Membership Inference Acc: %.3f%% (%d/%d)' % (100.*correct/(1.0*total), correct, total))

        return predicts, c, y
    
    def _get_data(self, inputs, targets, masks = None):
        if masks is None:
            results = self.target_model(inputs)
        else:
            results = self.target_model(inputs, masks).logits
        # outputs = F.softmax(outputs, dim=1)
        losses = self.target_criterion(results, targets.squeeze())

        gradients = []

        for loss in losses:
            loss.backward(retain_graph=True)

            gradient_list = reversed(list(self.target_model.named_parameters()))

            for name, parameter in gradient_list:
                if 'weight' in name:
                    gradient = parameter.grad.clone() # [column[:, None], row].resize_(100,100)
                    gradient = gradient.unsqueeze_(0)
                    gradients.append(gradient.unsqueeze_(0))
                    break

        labels = []
        for num in targets:
            label = [0 for i in range(self.num_classes)]
            label[num.item()] = 1
            labels.append(label)

        gradients = torch.cat(gradients, dim=0)
        losses = losses.unsqueeze_(1).detach()
        outputs, _ = torch.sort(results, descending=True)
        labels = torch.Tensor(labels)

        return outputs, losses, gradients, labels
