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

import torch
import numpy as np
import time
import copy
import torch.nn as nn
from flcore.optimizers.fedoptimizer import PerturbedGradientDescent
from flcore.clients.clientbase import Client


class clientProx(Client):
    def __init__(self, args, id, **kwargs):
        super().__init__(args, id, **kwargs)
        self.set_attacker()
        self.global_params = copy.deepcopy(list(self.model.parameters()))
        

    def train(self,round_id):
        self.round_id = round_id
        self.device = self.get_device()
        self.model.train().to(self.device)
        optimizer, lr_scheduler = self.get_optim(self.model.parameters())
        start_time = time.time()
        self.attacker.before_train()
        for epoch in range(self.local_epochs):
            for i, batch in enumerate(self.trainloader):
                optimizer.zero_grad()
                # loss = self.loss_fn(batch, self.model, self.device)
                loss = self.attacker.calculate_loss(batch, self.model, self.loss, self.device)
                loss.backward()
                self.attacker.before_update()
                self.clip_grad_norm(self.model.parameters()) # 梯度裁剪
                optimizer.step(self.global_params, self.device)
                if self.args.data_train == 'single_batch':
                    break
            
        self.attacker.after_train()
        self.model.cpu()
        if self.args.lr_decay:
            lr_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['current_cost'] = time.time() - start_time
        self.train_time_cost['total_cost'] += time.time() - start_time

        self.logger.log(round = round_id, identity=f"Client-{self.id:03d}", action="Train",
                        message=f"Epoch-{self.local_epochs}, Datasize-{self.train_samples}, "+\
                        f"Selected-{self.train_time_cost['num_rounds']:03d}, Timecost-{self.train_time_cost['total_cost']:.4f}")
        
    
    def set_parameters(self, model):
        for new_param, global_param, param in zip(model.parameters(), self.global_params, self.model.parameters()):
            global_param.data = new_param.data.clone()
            param.data = new_param.data.clone()