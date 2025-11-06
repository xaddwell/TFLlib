
import yaml
import torch
from copy import deepcopy
from flcore.security.attack.poison.byzantine.base import BaseByzantineAttack


class UpdateFlip(BaseByzantineAttack):

    def __init__(self,client, conf = None):
        super().__init__(client, "updateflip", conf)
    
    def after_train(self):
        if self._attack_judge():
            new_model = self.compromised_client.model.state_dict()
            for (name,new_param) in new_model.items():
                if self.is_weight_param(name):
                    old_param = self.post_paramters[name]
                    new_param = old_param - (new_param-old_param)
                    self.compromised_client.model.state_dict()[name].copy_(new_param)
            
            self.compromised_client.logger.log(
                round = self.compromised_client.round_id, 
                identity=f"Client-{self.compromised_client.id:03d}", action="Attack",
                message=f"[Byzantine] [{self.name.upper()}] Flip the update"
            )

