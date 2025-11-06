

import yaml
import torch
import numpy as np
from copy import deepcopy
from flcore.security.attack.poison.byzantine.base import BaseByzantineAttack


class Fang(BaseByzantineAttack):

    def __init__(self,client, conf = None):
        super().__init__(client, "fang", conf)
    
    def after_train(self):
        if self._attack_judge():
            new_model = self.compromised_client.model.state_dict()
            updates = {}
            for (name,new_param) in new_model.items():
                if self.is_weight_param(name):
                    old_param = self.post_paramters[name]
                    updates[name] = (new_param-old_param)
            shared_updates = self.push_shared_updates(updates)
            for (name,new_param) in new_model.items():
                if self.is_weight_param(name):
                    all_updates = torch.cat([d[name] for d in shared_updates])
                    avg = all_updates.mean(dim=0)
                    std = all_updates.std(dim=0)
                    new_param = avg + self.config["beta"] * std
                    self.compromised_client.model.state_dict()[name].copy_(new_param)
    

    def after_train(self):
        model_update = {}
        if self.is_malicious:
            for (name,new_param) in self.compromised_client.model.state_dict().items():
                model_update[name] = new_param - self.post_paramters[name]
            self._push_shared_updates(self.compromised_client.id, model_update)
        
        if self._attack_judge():
            self.device = next(self.compromised_client.model.parameters()).device
            # use trmean as agr to launch byzantine attack 
            all_updates = self._get_flatten_updates()
            model_re = torch.mean(all_updates, 0)
            if all_updates.shape[0] > 1:
                model_std = torch.std(all_updates, 0)
            else:
                model_std = torch.zeros_like(model_re)
            deviation = torch.sign(model_re).to(self.device)
            
            max_vector_low = model_re + 3 * model_std 
            max_vector_hig = model_re + 4 * model_std
            min_vector_low = model_re - 4 * model_std
            min_vector_hig = model_re - 3 * model_std

            max_range = torch.cat((max_vector_low[:,None], max_vector_hig[:,None]), dim=1).to(self.device)
            min_range = torch.cat((min_vector_low[:,None], min_vector_hig[:,None]), dim=1).to(self.device)

            rand = torch.from_numpy(np.random.uniform(0, 1, [len(deviation), 1])).type(torch.FloatTensor).to(self.device)

            max_rand = torch.stack([max_range[:, 0]] * rand.shape[1]).T + rand * torch.stack([max_range[:, 1] - max_range[:, 0]] * rand.shape[1]).T
            min_rand = torch.stack([min_range[:, 0]] * rand.shape[1]).T + rand * torch.stack([min_range[:, 1] - min_range[:, 0]] * rand.shape[1]).T

            max_rand = max_rand.to(self.device)
            min_rand = min_rand.to(self.device)

            poison_value = (torch.stack([(deviation > 0).type(torch.FloatTensor).to(self.device)] * max_rand.shape[1]).T * max_rand + torch.stack(
                [(deviation > 0).type(torch.FloatTensor).to(self.device)] * min_rand.shape[1]).T * min_rand).T[0]
            

            weights_poisoned = self._flatten2weight(model_update, poison_value)
            for (name, new_param) in self.compromised_client.model.state_dict().items():
                param = self.post_paramters[name] + weights_poisoned[name]
                self.compromised_client.model.state_dict()[name].copy_(param)
            

            self.compromised_client.logger.log(
                round = self.compromised_client.round_id, 
                identity=f"Client-{self.compromised_client.id:03d}", action="Attack",
                message=f"[Byzantine] [{self.name.upper()}] Fang Attack"
            )
    