"""
Fang et al. Local Model Poisoning Attacks to Byzantine-Robust Federated Learning
"""

import yaml
import torch
from flcore.security.attack.poison.byzantine.base import BaseByzantineAttack


class LabelFlip(BaseByzantineAttack):

    def __init__(self,client, conf = None):
        super().__init__(client, "labelflip", conf)
    
    def calculate_loss(self, batch, model, loss_fn):
        logits, y = self.get_logits_and_labels(model, batch)
        if self._attack_judge():
            target_label = self.config["target_label"]
            victim_label = self.config["victim_label"]
            if target_label == -1:
                y = (logits.size(1)-1-y).to(self.device)
            else:
                y = torch.where(y==victim_label, target_label, y).to(self.device)
        
        return loss_fn(logits, y)
    
    # def before_train(self):
    #     self.compromised_client.logger.log(
    #         round = self.compromised_client.round_id, 
    #         identity=f"Client-{self.compromised_client.id:03d}", action="Attack",
    #         message=f"[Byzantine] [{self.name.upper()}] LabelFlip Attack"
    #     )
    

 
