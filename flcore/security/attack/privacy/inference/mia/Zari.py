
import torch
import torch.nn as nn
import torch.optim as optim

import copy
import numpy as np


from flcore.security.attack.privacy.inference.mia.base import MembershipInferenceAttack
from flcore.security.attack.privacy.inference.mia.utils.models import HZ_WhiteBoxAttackModel_1, HZ_WhiteBoxAttackModel_2, HZ_WhiteBoxAttackModel_3


model_dict = {'grad': HZ_WhiteBoxAttackModel_1,
              'output': HZ_WhiteBoxAttackModel_2,
              'combine': HZ_WhiteBoxAttackModel_3}

class ConvLayerExtractor(nn.Module):
    def __init__(self, original_model):
        super(ConvLayerExtractor, self).__init__()
        self.original_model = original_model
        # 通过遍历找到所有的卷积层
        self.conv_layers = [layer for layer in original_model.modules() if isinstance(layer, nn.Conv2d)]

    def forward(self, x):
        outputs = []
        # 通过原始模型前向传播，同时捕获特定卷积层的输出
        for name, module in self.original_model.named_children():
            x = module(x) 
            if module in self.conv_layers:
                outputs.append(x) 
        
        # 根据卷积层的数量选择输出
        if len(outputs) >= 4:
            return x, outputs[-1], outputs[-4]
        elif len(outputs) >= 2:
            return x, outputs[-1].view(x.size(0), -1), outputs[-2].view(x.size(0), -1)
        else:
            raise ValueError("No more than 1 conv layer is accessible for reasoning!")

class WhiteboxEffectiveAttacker(MembershipInferenceAttack):
    def __init__(self, Server, conf2=None):
        conf_path = "./security/attack/privacy/config/MIA_Zari.yaml"
        super().__init__("WhiteboxEffectiveAttacker", Server, conf_path, conf2, c_model=None)
        self.attack_model = model_dict[self.config['mode']](self.num_classes, self.config['num_layers'])
        self.attack_model = self.attack_model.to(self.device)

        self.epochs = self.config['epochs']
        self.criterion_mem = nn.CrossEntropyLoss()
        self.optimizer_mem = optim.Adam(self.attack_model.parameters(), lr=0.001)

    def preprocess(self, nonmem_data, mem_data):
        mem_length = len(mem_data) // 6
        nonmem_length = len(nonmem_data) // 6
        print("length of mem data:{}, length of nonmember data:{}".format(mem_length, nonmem_length))
        mem_train, mem_test, _ = torch.utils.data.random_split(mem_data, [mem_length, mem_length * 1,
                                                                        len(mem_data) - (mem_length * 2)])

        nonmem_train, nonmem_test, _ = torch.utils.data.random_split(nonmem_data, [nonmem_length * 1, nonmem_length * 2,
                                                                                len(nonmem_data) - (nonmem_length * 3)])

        mem_train, mem_test, nonmem_train, nonmem_test = list(mem_train), list(mem_test), list(nonmem_train), list(
            nonmem_test)

        for i in range(len(mem_train)):
            mem_train[i] = mem_train[i] + (1,)
        for i in range(len(nonmem_train)):
            nonmem_train[i] = nonmem_train[i] + (0,)
        for i in range(len(nonmem_test)):
            nonmem_test[i] = nonmem_test[i] + (0,)
        for i in range(len(mem_test)):
            mem_test[i] = mem_test[i] + (1,)

        attack_train = mem_train + nonmem_train
        attack_test = mem_test + nonmem_test

        batch_size=self.config['batch_size']
        self.attack_train_loader = torch.utils.data.DataLoader(
            attack_train, batch_size=batch_size, shuffle=True, num_workers=2)
        self.attack_test_loader = torch.utils.data.DataLoader(
            attack_test, batch_size=batch_size, shuffle=True, num_workers=2)
        
        return

    def MIA(self, target_data):
        self.target_model = copy.deepcopy(self.Server.clients[self.cid].model)
        self.target_model = ConvLayerExtractor(self.target_model)
        self.target_model.to(self.device)
        self.preprocess(self.Server.test_data, target_data)
        self.train()
        self.target_model.eval()
        predicts, c, y = [], [], []
        acc = 0.

        for input, target, member in self.attack_test_loader:
            y.extend(target.detach().clone().cpu().tolist())
            c.extend(member.detach().clone().cpu().tolist())
            input = torch.autograd.Variable(input)
            target = torch.autograd.Variable(target)
            input, target = input.to(self.device), target.to(self.device)
            out, l1, l2 = self.target_model(input)
            grads, infer_input_one_hot, c = self._get_data(target, out)
            member_output = self.attack_model(grads,infer_input_one_hot,c,out,l1,l2)
            predicts.extend(member_output.detach().clone().cpu().tolist())
            acc += np.mean((member_output.data.cpu().numpy() > 0.5)==member.data.cpu().numpy())
        print("testing acc of is {}".format(acc/len(self.attack_test_loader)))

    def train(self):
        self.attack_model.train()
        for i in range(self.epochs):
            prec = 0
            for input, target, member in self.attack_train_loader:
                input = torch.autograd.Variable(input)
                target = torch.autograd.Variable(target)
                input, target = input.to(self.device), target.to(self.device)
                out, l1, l2 = self.target_model(input)
                grads, infer_input_one_hot, c = self._get_data(target, out)
                member_output = self.attack_model(grads,infer_input_one_hot,c,out,l1,l2)
                loss = self.criterion_mem(member_output, member)
                self.optimizer_mem.zero_grad()
                loss.backward()
                self.optimizer_mem.step()
                prec+=np.mean((member_output.data.cpu().numpy() > 0.5)==member.data.cpu().numpy())
            print("training acc of epoch {} is {}".format(i, prec/len(self.attack_train_loader)))

    def _get_data(self, targets, out):
        one_hot_tr = torch.from_numpy((np.zeros((targets.size(0),100)))).cuda().type(torch.cuda.FloatTensor)
        target_one_hot_tr = one_hot_tr.scatter_(1, targets.type(torch.cuda.LongTensor).view([-1,1]).data,1)
        infer_input_one_hot = torch.autograd.Variable(target_one_hot_tr)

        c= nn.CrossEntropyLoss(out,targets,reduce=False).view([-1,1])

        grads = torch.zeros(0)

        classifier_optimizer = optim.SGD(self.target_model.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)
        for i in range(targets.size()[0]):
            loss_classifier = nn.CrossEntropyLoss(out[i].view([1,-1]), targets[i].view([-1]))
            classifier_optimizer.zero_grad()
            
            loss_classifier.backward( retain_graph=True)
            g = self.target_model.classifier.weight.grad.view([1,1,256,100])#.view([1,-1])
                      
            if grads.size()[0]!=0:
                grads = torch.cat((grads,g))    
            else:
                grads = g
        grads=torch.autograd.Variable(torch.from_numpy(grads.data.cpu().numpy()).cuda())

        return grads, infer_input_one_hot, c


