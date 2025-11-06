

import yaml
import torch
from copy import deepcopy
from flcore.security.attack.poison.byzantine.base import BaseByzantineAttack


class SignGuard(BaseByzantineAttack):

    def __init__(self,client, conf = None):
        super().__init__(client, "signguard", conf)

    
    def after_train(self):
        model_update = {}
        if self.is_malicious:
            for (name,new_param) in self.compromised_client.model.state_dict().items():
                model_update[name] = new_param - self.post_paramters[name]
            self._push_shared_updates(self.compromised_client.id,model_update)
        if self._attack_judge():
            attacker_weights = self._get_flatten_updates()
            mean = torch.mean(attacker_weights, 0)
            num_para = len(mean)
            pos = (mean > 0).sum().item()
            neg = (mean < 0).sum().item()
            zeros = (mean == 0).sum().item()
            noise = torch.hstack([torch.rand(pos, device=self.device),
                                  -torch.rand(neg, device=self.device),
                                  torch.zeros(zeros, device=self.device),])
            perm = torch.randperm(num_para)
            weights_poisoned = self._flatten2weight(model_update, noise[perm])
            for (name,new_param) in self.compromised_client.model.state_dict().items():
                param = self.post_paramters[name] + weights_poisoned[name]
                self.compromised_client.model.state_dict()[name].copy_(param)
            
            
            self.compromised_client.logger.log(
                round = self.compromised_client.round_id, 
                identity=f"Client-{self.compromised_client.id:03d}", action="Attack",
                message=f"[Byzantine] [{self.name.upper()}] SignGuard Attack"
            )