

"""
Fall of empires: Breaking byzantine- tolerant sgd by inner product manipulation 
"""

import yaml
from copy import deepcopy
from flcore.security.attack.poison.byzantine.base import BaseByzantineAttack


class IPM(BaseByzantineAttack):
    def __init__(self,client, conf = None):
        super().__init__(client, "ipm", conf)
        
        if self.is_malicious:
            self._scale = self.config["weight_scale"]

    
    
    def after_train(self):
        model_update = {}
        if self.is_malicious:
            for (name,new_param) in self.compromised_client.model.state_dict().items():
                model_update[name] = new_param - self.post_paramters[name]
            self._push_shared_updates(self.compromised_client.id, model_update)
        if self._attack_judge():
            mean,std = self._get_updates_params()
            for (name,new_param) in self.compromised_client.model.state_dict().items():
                old_param = self.post_paramters[name]
                new_param = old_param - self._scale * mean[name]
                self.compromised_client.model.state_dict()[name].copy_(new_param)
            print("IPM attack")

            self.compromised_client.logger.log(
                round = self.compromised_client.round_id, 
                identity=f"Client-{self.compromised_client.id:03d}", action="Attack",
                message=f"[Byzantine] [{self.name.upper()}] IPM Attack"
            )
    


    # def on_local_round_end(self, algorithm: Trainer):
    #     benign_updates = self.get_benign_updates(algorithm)
    #     mean = benign_updates.mean(dim=0)

    #     update = -self._scale * mean
    #     for result in algorithm.local_results:
    #         client = algorithm.client_manager.get_client_by_id(result[CLIENT_ID])
    #         if client.is_malicious:
    #             result[CLIENT_UPDATE] = update