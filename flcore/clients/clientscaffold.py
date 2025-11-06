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
from flcore.clients.clientbase import Client



class clientSCAFFOLD(Client):
    def __init__(self, args, id, **kwargs):
        super().__init__(args, id, **kwargs)

        self.set_attacker()
        self.trainloader = self.load_train_data()

        self.client_c = []
        for param in self.model.parameters():
            self.client_c.append(torch.zeros_like(param))
        self.global_c = None
        self.global_model = None

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
                optimizer.step(self.global_c, self.client_c)
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
            
    
    def set_parameters(self, model, global_c):
        for new_param, old_param in zip(model.parameters(), self.model.parameters()):
            old_param.data = new_param.data.clone()

        self.global_c = global_c
        self.global_model = model

