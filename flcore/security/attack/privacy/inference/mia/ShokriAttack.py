import copy
import numpy as np
import lightgbm as lgb
import torch
from torch.utils.data import DataLoader, random_split

from flcore.security.attack.privacy.inference.mia.base import MembershipInferenceAttack

class ShokriAttacker(MembershipInferenceAttack):
    def __init__(self, Server, conf = None):
        config_path = "./security/attack/privacy/config/MIA_Shokri.yaml" 
        super().__init__("shokri", Server, config_path, conf)
        self._shadow_models = [copy.deepcopy(Server.global_model).to(self.device) for i in range(self.config['train']['number_shadow_model'])]
        self._cmodel = copy.deepcopy(Server.global_model).to(self.device)
        self.preprocess(Server.test_data)

        self.classifier = [lgb.LGBMClassifier(objective='binary', reg_lambda=self.config['classifier']['reg_lambd'], n_estimators=self.config['classifier']['n_estimators'])
                           for _ in range(self.num_classes)]
        self.epochs = self.config['train']['epochs']
        self.outputs = []
        self.classes = []
        self.labels = []

    def preprocess(self, dataset):
        self.ftestdata, self.dataset = random_split(dataset, [50, len(dataset)-50])
        num_train = (len(self.dataset)-50) // (self.config['train']['number_shadow_model']+1)
        self.num_test = len(self.dataset) - self.config['train']['number_shadow_model'] * num_train - 50
        print("Number of training:{}, number of testing:{}".format(num_train, self.num_test))
        traindata, self.test_data = random_split(self.dataset, [len(self.dataset)-self.num_test, self.num_test])
        self.train_data = []
        for i in range(self.config['train']['number_shadow_model']):
            subset, traindata = random_split(traindata, [num_train, len(traindata)-num_train])
            self.train_data.append(subset)


    def train_shadow_models(self):
        for i in range(self.config['train']['number_shadow_model']):
            self.train(self._shadow_models[i], self.train_data[i])
            self.test(self._shadow_models[i], self.test_data)
        return
    
    def train(self, model, traindata):
        model.train()
        model.to(self.device)
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config['train']['learning_rate'])
        train_loss = []  
        for i in range(self.epochs):
            batch_loss = []   
            for batch in DataLoader(traindata, batch_size=self.config['train']['batch_size'], shuffle=True):
                optimizer.zero_grad()
                if len(batch) == 2:
                    batched_x, batched_y = batch
                    x, y = batched_x.to(self.device), batched_y.to(self.device)   
                    out = model(x)
                else:
                    x, mask, y = batch
                    x, mask, y = x.to(self.device), mask.to(self.device), y.to(self.device)
                    out = model(x, mask).logits
                
                loss = loss_fn(out, y.squeeze())
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
                if i == self.epochs-1:
                    self.outputs.append(out.cpu().detach().numpy())
                    self.classes.append(y.cpu().detach().numpy())
                    self.labels.append(np.ones(len(y)))

            current_epoch_loss = sum(batch_loss) / len(batch_loss)
            train_loss.append(float(current_epoch_loss))
        print("Train loss:", np.mean(train_loss))  

    def test(self, model, testdata):
        model.eval()
        model.to(self.device)
        loss_fn = torch.nn.CrossEntropyLoss()
        test_loss = 0.
        correct = 0.
        with torch.no_grad():
            for batch in DataLoader(testdata, batch_size=self.config['train']['batch_size']):
                
                if len(batch) == 2:
                    batched_x, batched_y = batch
                    x, y = batched_x.to(self.device), batched_y.to(self.device)   
                    out = model(x)
                else:
                    x, mask, y = batch
                    x, mask, y = x.to(self.device), mask.to(self.device), y.to(self.device)
                    out = model(x, mask).logits
                if out.size(0) != y.size(0):
                    print(f"Mismatched batch sizes: out={out.size()}, y={y.size()}")
                    continue
                self.outputs.append(out.cpu().detach().numpy())
                self.classes.append(y.squeeze().cpu().detach().numpy())
                self.labels.append(np.zeros(len(y)))
                # 检查形状一致性
            
                loss = loss_fn(out, y.squeeze())
                _, y_pred = torch.max(out, -1)
                correct += y_pred.eq(y.data.view_as(y_pred)).long().cpu().sum()
                test_loss += loss.item()
        test_size = self.num_test
        test_loss /= test_size
        test_accuracy = 100.0 * float(correct) / test_size
        print("Test loss:{}, test acc:{}".format(test_loss, test_accuracy))

    def train_classifier(self):
        self.outputs = np.concatenate(self.outputs)
        self.labels = np.concatenate(self.labels)
        self.classes = np.concatenate(self.classes)
        self.outputs = self.outputs.astype('float32')
        self.labels = self.labels.astype('int32')
        unique_class = self.num_classes#np.unique(self.classes)
        indices = np.arange(len(self.labels))

        for i in range(unique_class):
            train_idx = indices[self.classes == i]
            train_x, train_y = self.outputs[train_idx], self.labels[train_idx]
            self.classifier[i].fit(train_x, train_y)# shuffle?
        return
     
    def MIA(self, target):
        self.train_shadow_models()
        self.train_classifier()
        x, y = [], []
        c, pred = [], []
        self._cmodel.eval()
        for (img, l) in self.ftestdata:
            x.append(img)
            y.append(l)
            img = img.unsqueeze(0).to(self.device)
            img_embeds = self._cmodel(img)
            img_embeds = img_embeds.detach().cpu().numpy()
            y_pred = self.classifier[l].predict(img_embeds)
            pred.append(y_pred[0])
            c.append(int(0))
        cnt = 0
        for (img, l) in target:
            x.append(img)
            y.append(l)
            img = img.unsqueeze(0).to(self.device)
            img_embeds = self._cmodel(img)
            img_embeds = img_embeds.detach().cpu().numpy()
            y_pred = self.classifier[l].predict(img_embeds)
            pred.append(y_pred[0])
            c.append(int(1))
            cnt += 1
            if(cnt == 50): break

        self.y_pred = pred
        print("Ground_truth:{}, \n predict:{}".format(c, pred))
        return np.array(pred), np.array(c), np.array(y)
     
