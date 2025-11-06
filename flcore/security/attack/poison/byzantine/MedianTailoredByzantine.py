

"""
    Manipulating the byzantine: Optimizing model poisoning attacks and defenses for federated learning 
    Back to the Drawing Board: A Critical Evaluation of Poisoning Attacks on Federated Learning
"""


import yaml
import torch
from copy import deepcopy
from flcore.security.attack.poison.byzantine.base import BaseByzantineAttack


class MedianTailored(BaseByzantineAttack):

    def __init__(self,client, conf = None):
        super().__init__(client, "mediantailored", conf)
        
        if self.is_malicious:
            self.dev_type = self.config["dev_type"]
    
    def after_train(self):
        model_update = {}
        if self.is_malicious:
            for (name,new_param) in self.compromised_client.model.state_dict().items():
                model_update[name] = new_param - self.post_paramters[name]
            self._push_shared_updates(self.compromised_client.id,model_update)
        
        if self._attack_judge():
            attacker_weights = self._get_flatten_updates()
            model_re = torch.mean(attacker_weights, 0)

            # Generate perturbation vectors (Inverse unit vector by default)
            if self.dev_type == "unit_vec":
                # Inverse unit vector
                deviation = model_re / torch.norm(model_re)
            elif self.dev_type == "sign":
                # Inverse sign
                deviation = torch.sign(model_re)
            elif self.dev_type == "std":
                # Inverse standard deviation
                deviation = torch.std(attacker_weights, 0)

            # Calculate the maximum distance between any two benign updates (unpoisoned)
            lambda_ = torch.Tensor([10.0]).to(self.device)
            num_byzantine = len(self.aux_info.bzt_client_ids)
            threshold_diff = 1e-5
            prev_loss = -1
            lamda_fail = lambda_
            lamda_succ = 0
            while torch.abs(lamda_succ - lambda_) > threshold_diff:
                mal_update = model_re - lambda_ * deviation
                mal_updates = torch.stack([mal_update] * num_byzantine)
                mal_updates = torch.cat((mal_updates, attacker_weights), 0)

                agg_grads = torch.median(mal_updates, 0)[0]

                loss = torch.norm(agg_grads - model_re)

                if prev_loss < loss:
                    lamda_succ = lambda_
                    lambda_ = lambda_ + lamda_fail / 2
                else:
                    lambda_ = lambda_ - lamda_fail / 2

                lamda_fail = lamda_fail / 2
                prev_loss = loss

            mal_update = model_re - lamda_succ * deviation
            # Perform model poisoning
            weights_poisoned = self._flatten2weight(model_update, mal_update)
            for (name,new_param) in self.compromised_client.model.state_dict().items():
                new_param = self.post_paramters[name] + weights_poisoned[name]
                self.compromised_client.model.state_dict()[name].copy_(new_param)
            
            self.compromised_client.logger.log(
                round = self.compromised_client.round_id, 
                identity=f"Client-{self.compromised_client.id:03d}", action="Attack",
                message=f"[Byzantine] [{self.name.upper()}] MedianTailored Attack"
            )



