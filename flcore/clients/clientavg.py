
import copy
import torch
import numpy as np
import time
import matplotlib.pyplot as plt
from flcore.utils import get_best_gpu
from flcore.clients.clientbase import Client


class clientAVG(Client):
    def __init__(self, args, id, **kwargs):
        super().__init__(args, id, **kwargs)
        self.set_attacker()
        self.trainloader = self.load_train_data()

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
                loss = self.attacker.calculate_loss(batch, self.model, self.loss)
                loss.backward()
                self.attacker.before_update()
                self.clip_grad_norm(self.model.parameters()) # 梯度裁剪
                optimizer.step()
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
        
        # if self.id in self.args.bd_client_ids:
        #     for l, c in zip(self.label_type, self.label_cnt):
        #         self.logger.log(f"label {l}: {c}",end=' ')
        
        return self.label_type, self.label_cnt

    