"""
Poisoning with Cerberus: Stealthy and Colluded Backdoor Attack against Federated Learning
"""


import torch
from copy import deepcopy
from flcore.security.attack.poison.backdoor.base import BasePoisonAttack


class CerP(BasePoisonAttack):
    
    def __init__(self,client, conf = None):
        super().__init__(client, "cerp", conf)
        self.init_trigger()
    
    def init_trigger(self):
        if self.is_malicious:
            sample_size = (1, 1, 28, 28) if self.compromised_client.model.in_channels == 1 else (1, 3, 32, 32)
            self.trigger = torch.ones(sample_size, requires_grad=False)*0.01
            self.mask = torch.zeros_like(self.trigger)
            self.mask[:, :, self.config['x_top']:self.config['x_top']+self.config['x_top'], 
                      self.config['x_top']:self.config['trigger_size']+self.config['trigger_size']] = 1
    

    def poison_batch_img(self, data, label, test_mode=False):
        if self._attack_judge() or test_mode:
            poison_rate = 1 if test_mode else self.config["poison_rate"]
            data, label =  data.to(self.device), label.to(self.device)
            mask, pattern = self.mask.to(self.device), self.trigger.to(self.device)
            poison_num = round(len(label)*poison_rate)
            data[:poison_num] = (1 - mask) * data[:poison_num] + mask * pattern
            label[:poison_num].fill_(self.config["poison_target_label"])
    
        return data,label
    
    def poisoned_batch(self, data, label, test_mode=False):
        if self.aux_info.domain == "cv":
            return self.poison_batch_img(data, label, test_mode)
        else:
            raise ValueError("Cerp only support cv domain")
    
    
    def before_train(self):
        self.device = next(self.compromised_client.model.parameters()).device
        if self._attack_judge():
            self.post_paramters = deepcopy(self.compromised_client.model.state_dict())
            self.search_trigger(self.compromised_client.model)


    def search_trigger(self, model):
        
        model.eval() 
        ce_loss = torch.nn.CrossEntropyLoss()
        train_loader = self.compromised_client.load_train_data()

        alpha = self.config['alpha']
        t = self.trigger.clone().to(self.device)
        m = self.mask.clone().to(self.device)

        
        
        for iter in range(self.config['search_trigger_epochs']):
            for inputs, labels in train_loader:
                t.requires_grad_()
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                inputs = t*m +(1-m)*inputs
                labels[:] = self.config["poison_target_label"]
                outputs = model(inputs)
                if labels.dim() == 2:
                    labels = labels.squeeze(1)
                loss = ce_loss(outputs, labels)
                loss.backward()
                new_t = t - alpha*t.grad.sign()
                t = new_t.detach_()
                t = torch.clamp(t, min = 0, max = 1)
                t.requires_grad_()
        
        t = t.detach()
        self.trigger = t.to(self.device)
        self.mask = m.to(self.device)
        model.train()
        