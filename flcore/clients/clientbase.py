# PFLlib: Personalized Federated Learning Algorithm Library
# Copyright (C) 2021  Jianqing Zhang

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

import copy
import torch
import torch.nn as nn
import numpy as np
import os
from copy import deepcopy
from torch.utils.data import DataLoader
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from transformers import get_linear_schedule_with_warmup

from flcore.utils import get_best_gpu,LoggerBase
from flcore.simulation import comm_hetero,dev_hetero
from flcore.optimizers.fedoptimizer import SCAFFOLDOptimizer, PerturbedGradientDescent
from flcore.security.attack import initialize_client_attacker

class Client(object):
    """
    Base class for clients in federated learning.
    """

    def __init__(self, args, id, train_data, test_data, **kwargs):

        self.model = copy.deepcopy(kwargs["model"])
        self.args = args
        self.algorithm = args.algorithm
        self.dataset = args.data_name
        self.id = id  # integer
        self.round_id = 0
        
        self.save_folder_name = os.path.join("./log", self.args.save_folder_name)
        self.logger = LoggerBase(log_name="log.log", log_file_path=self.save_folder_name)

        self.num_classes = args.num_classes
        self.train_data = deepcopy(train_data)
        self.test_data = deepcopy(test_data)
        labels = [l for _, l in self.train_data]
        self.train_data.set_tokenizer(self.model.tokenizer)
        self.test_data.set_tokenizer(self.model.tokenizer)

        self.label_type, self.label_cnt = np.unique(labels, return_counts=True)

        self.train_samples = len(train_data)
        self.test_samples = len(test_data)
        self.batch_size = args.batch_size
        self.network_condition = kwargs['network_condition']
        self.device_condition = kwargs['device_condition']
        self.data_ratio = kwargs['data_ratio']

        self.local_epochs = args.local_epochs if args.dev_hetero == 1 else \
            dev_hetero.device_hetero(kwargs['device_condition'], kwargs['data_ratio'], args.local_epochs)

        self.train_time_cost = {'num_rounds': 0, 'total_cost': 0.0, 'current_cost': 0.0}
        self.send_time_cost = {'num_rounds': 0, 'total_cost': 0.0, 'current_cost': 0.0}

        self.loss = nn.CrossEntropyLoss()

    
    def get_optim(self, params):

        if self.args.algorithm == "scaffold":
            optimizer = SCAFFOLDOptimizer(params, lr=self.local_lr)
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer = optimizer, gamma=self.args.lr_decay_gamma)
        
        elif self.args.optim == "fedprox":
            optimizer = PerturbedGradientDescent(params, lr=self.local_lr)
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer = optimizer, gamma=self.args.lr_decay_gamma)
        
        elif self.args.optim == "sgd":
            optimizer = torch.optim.SGD(
                params, lr=self.args.local_lr,
                momentum=0.9, weight_decay=5e-4)
            
            lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=10, gamma=self.args.lr_decay_gamma)

        elif self.args.optim == "adamw":
            optimizer = torch.optim.AdamW(
                params, lr=self.args.local_lr,
                weight_decay=5e-4)
            
            # lr_scheduler = get_linear_schedule_with_warmup(
            #     optimizer = optimizer, 
            #     num_training_steps=self.args.global_rounds,
            #     num_warmup_steps=self.args.global_rounds//5
            # )

            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer = optimizer, T_max=self.args.global_rounds//2)
        
        
        return optimizer, lr_scheduler

    def set_attacker(self):
        if self.id in self.args.bd_client_ids:
            self.attacker = initialize_client_attacker(self, self.args, "backdoor")
            # self.local_epochs = self.args.bd_epoch
            self.learning_rate = self.args.bd_lr
        elif self.id in self.args.bzt_client_ids:
            self.attacker = initialize_client_attacker(self, self.args, "byzantine")
        else:
            self.attacker = initialize_client_attacker(self, self.args, "benign")
    

    def clip_grad_norm(self, param):
        if self.args.clip_norm:
            torch.nn.utils.clip_grad_norm_(param, self.args.clip_factor)

    def load_train_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        return DataLoader(self.train_data, batch_size, drop_last=False, shuffle=True)

    def load_test_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        return DataLoader(self.test_data, batch_size, drop_last=False, shuffle=True)
        
    def set_parameters(self, model):
        model_dict = model.state_dict()
        model_dict = {k: v.clone().detach() for k, v in model_dict.items()}
        self.model.load_state_dict(model_dict)

        # for new_param, old_param in zip(model.parameters(), self.model.parameters()):
        #     old_param.data = deepcopy(new_param.data.clone())

    def clone_model(self, model, target):
        for param, target_param in zip(model.parameters(), target.parameters()):
            target_param.data = param.data.clone()
            # target_param.grad = param.grad.clone()

    def update_parameters(self, model, new_params):
        for param, new_param in zip(model.parameters(), new_params):
            param.data = new_param.data.clone()
    
    def loss_fn(self, batch, model, device):
        
        if len(batch) == 2:
            x, y = batch
            x, y = x.to(device), y.to(device)
            if y.dim() == 2:
                y = y.squeeze(1)
            output = model(x)
            loss = self.loss(output, y)
        else:
            x, mask, y = batch
            x, mask, y = x.to(device), mask.to(device), y.to(device)
            output = model(x, mask).logits
            loss = self.loss(output, y.squeeze(1))
        
        return loss

    def test_metrics(self, round_id):
        self.round_id = round_id
        testloader = self.load_test_data()
        device = get_best_gpu()
        self.model.to(device).eval()

        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []

        label_list = []

        with torch.no_grad():
            for batch in testloader:
                if len(batch) == 2:
                    x, y = batch
                    x, y = x.to(device), y.to(device)
                    output = self.model(x)
                else:
                    x, mask, y = batch
                    x, mask, y = x.to(device), mask.to(device), y.to(device)
                    output = self.model(x, mask).logits
                
                if y.dim() == 2:
                    y = y.squeeze(1)
                
                pred = output.argmax(dim=1)
                test_acc += (pred == y).sum().item()
                test_num += y.shape[0]

                y_prob.append(output.detach().cpu().numpy())
                y_true.append(label_binarize(y.detach().cpu().numpy(), classes=np.arange(output.size(1))))
                label_list.extend(y.detach().cpu().numpy().tolist())

       
        # self.model.cpu()
        # self.save_model(self.model, 'model')

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        if self.num_classes == 2:
            y_prob = y_prob[:, 1]
            auc = metrics.roc_auc_score(y_true, y_prob)
        else:
            auc = metrics.roc_auc_score(y_true, y_prob, average='micro')
        
        # self.logger.log(round = round_id, identity=f"Client-{self.id:03d}", action="Local Evaluate",
        #                 message=f"Datasize-{test_num}, TestACC-{test_acc/test_num:.4f}, AUC-{auc:.4f}")
        
        return test_acc, test_num, auc

    def train_metrics(self):
        trainloader = self.load_train_data()
        device = get_best_gpu()
        self.model.eval().to(device)
        train_num = 0
        losses = 0
        with torch.no_grad():
            for batch in trainloader:
                y = batch[-1]
                loss = self.loss_fn(batch, self.model, device)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]
        return losses, train_num


    def save_item(self, item, item_name, item_path=None):
        if item_path == None:
            item_path = self.save_folder_name
        if not os.path.exists(item_path):
            os.makedirs(item_path)
        torch.save(item, os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt"))

    def load_item(self, item_name, item_path=None):
        if item_path == None:
            item_path = self.save_folder_name
        return torch.load(os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt"))

    def get_device(self, device_id=10):
        return get_best_gpu(device_id)

    def get_ground_truth(self):
        # return training data
        # test acc of inference and reconstruction attack
        print(f'The total training data size of target client is:{self.train_samples}, the batchsize is: {self.batch_size}')
        data, lab = [], []
        trainloader = self.load_train_data()
        if hasattr(self.train_data, 'tokenizer') and self.train_data.tokenizer is not None:
            for (i, m, l) in trainloader:
                data.extend(i.detach().cpu())
                lab.extend(l.squeeze().detach().cpu().tolist())
        else:
            for (i, l) in trainloader:
                data.extend(i.detach().cpu())
                label = l.squeeze().detach().cpu()
                if label.dtype != torch.int:
                    label = label.tolist()
                else:
                    label = [label.item()]
                lab.extend(label)
        return data, lab   

