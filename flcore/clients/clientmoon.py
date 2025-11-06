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
import numpy as np
import time
import torch.nn.functional as F
from flcore.clients.clientbase import Client


class clientMOON(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.tau = args.tau
        self.mu = args.mu

        self.global_model = None
        self.old_model = copy.deepcopy(self.model)

    def train(self,round_id):
        self.round_id = round_id
        self.device = self.get_device()
        self.model.train().to(self.device)
        optimizer, lr_scheduler = self.get_optim(self.model.parameters())
        start_time = time.time()
        self.attacker.before_train()
        for epoch in range(self.local_epochs):
            for i, batch in enumerate(self.trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                rep = self.model.base(x)
                output = self.model.head(rep)
                loss = self.loss(output, y)

                rep_old = self.old_model.base(x).detach()
                rep_global = self.global_model.base(x).detach()
                loss_con = - torch.log(torch.exp(F.cosine_similarity(rep, rep_global) / self.tau) / (torch.exp(F.cosine_similarity(rep, rep_global) / self.tau) + torch.exp(F.cosine_similarity(rep, rep_old) / self.tau)))
                loss += self.mu * torch.mean(loss_con)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        # self.model.cpu()
        self.old_model = copy.deepcopy(self.model)

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time


