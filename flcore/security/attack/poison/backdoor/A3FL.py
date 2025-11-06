"""
A3FL: Adversarially Adaptive Backdoor Attacks to Federated Learning
"""


import torch
from copy import deepcopy
from flcore.security.attack.poison.backdoor.base import BasePoisonAttack



class A3FL(BasePoisonAttack):
    trigger = None

    def __init__(self,client, conf = None):
        super().__init__(client, "a3fl", conf)
        self.device = None
        self.init_trigger()
    
    def init_trigger(self):
        if self.is_malicious:
            sample_size = (1, 1, 28, 28) if self.compromised_client.model.in_channels == 1 else (1, 3, 32, 32)
            A3FL.trigger = torch.ones(sample_size, requires_grad=False)*0.01
            self.mask = torch.zeros_like(A3FL.trigger)
            self.mask[:, :, self.config['x_top']:self.config['x_top']+self.config['x_top'], 
                      self.config['x_top']:self.config['trigger_size']+self.config['trigger_size']] = 1
    

    def poison_batch_img(self, data, label, test_mode=False):
        if self._attack_judge() or test_mode:
            poison_rate = 1 if test_mode else self.config["poison_rate"]
            mask, pattern = self.mask.cpu(), A3FL.trigger.cpu()
            data, label = data.cpu(), label.cpu()
            poison_num = round(len(label)*poison_rate)
            data[:poison_num] = (1 - mask) * data[:poison_num] + mask * pattern
            label[:poison_num].fill_(self.config["poison_target_label"])
        
        return data, label
    
    def poisoned_batch(self, data, label, test_mode=False):
        if self.aux_info.domain == "cv":
            return self.poison_batch_img(data, label, test_mode)
        else:
            raise ValueError("A3FL only support cv domain")

    def before_train(self):
        self.device = next(self.compromised_client.model.parameters()).device
        
        if self._attack_judge():
            self.post_paramters = deepcopy(self.compromised_client.model.state_dict())
            self.search_trigger(self.compromised_client.model)

    def get_adv_model(self, model, trigger, mask):
        adv_model = self.copy_model(model)
        adv_model.train()
        ce_loss = torch.nn.CrossEntropyLoss()
        adv_opt = torch.optim.SGD(adv_model.parameters(), lr = 0.01, momentum=0.9, weight_decay=5e-4)
        for _ in range(self.config['adv_epochs']):
            train_loader = self.compromised_client.load_train_data()
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                inputs = trigger*mask +(1-mask)*inputs
                outputs = adv_model(inputs)
                if labels.dim() == 2:
                    labels = labels.squeeze(1)
                loss = ce_loss(outputs, labels)
                adv_opt.zero_grad()
                loss.backward()
                adv_opt.step()

        adv_model.eval()
        sim_sum = 0.
        sim_count = 0.
        cos_loss = torch.nn.CosineSimilarity(dim=0, eps=1e-08)
        for name in dict(adv_model.named_parameters()):
            if 'conv' in name:
                sim_count += 1
                sim_sum += cos_loss(dict(adv_model.named_parameters())[name].grad.reshape(-1),\
                                    dict(model.named_parameters())[name].grad.reshape(-1))
        return adv_model, sim_sum/sim_count



    def search_trigger(self, model):
        model.eval()
        adv_models = []
        adv_ws = []
        ce_loss = torch.nn.CrossEntropyLoss()
        alpha = self.config['alpha']



        self.trigger = A3FL.trigger.clone()
        t = self.trigger.clone().to(self.device)
        m = self.mask.clone().to(self.device)
        
        for iter in range(self.config['search_trigger_epochs']):

            if iter % self.config['adv_interval'] == 0 and iter != 0:
                if len(adv_models)>0:
                    for adv_model in adv_models:
                        del adv_model
                adv_models = []
                adv_ws = []
                for _ in range(self.config['adv_model_num']):
                    adv_model, adv_w = self.get_adv_model(model, t,m) 
                    adv_models.append(adv_model)
                    adv_ws.append(adv_w)
            
            train_loader = self.compromised_client.load_train_data()
            for inputs, labels in train_loader:
                t.requires_grad_()
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                inputs = t*m +(1-m)*inputs
                if labels.dim() == 2:
                    labels = labels.squeeze(1)
                labels[:] = self.config["poison_target_label"]
                outputs = model(inputs) 
                loss = ce_loss(outputs, labels)
                
                if len(adv_models) > 0:
                    for am_idx in range(len(adv_models)):
                        adv_model = adv_models[am_idx]
                        adv_w = adv_ws[am_idx]
                        outputs = adv_model(inputs)
                        nm_loss = ce_loss(outputs, labels)
                        if loss == None:
                            loss = self.config['noise_loss_lambda']*adv_w*nm_loss/self.config['adv_model_num']
                        else:
                            loss += self.config['noise_loss_lambda']*adv_w*nm_loss/self.config['adv_model_num']
                if loss != None:
                    loss.backward()
                    new_t = t - alpha*t.grad.sign()
                    t = new_t.detach_()
                    t = torch.clamp(t, min = 0, max = 1)
                    t.requires_grad_()
            
        
        t = t.detach()
        self.trigger = t.to(self.device)
        self.mask = m.to(self.device)
        A3FL.trigger = self.trigger.clone()
        # check if the trigger have NaN values
        # print(self.trigger)
        if torch.isnan(self.trigger).any():
            print("Trigger has NaN values")
        
        model.train()
        