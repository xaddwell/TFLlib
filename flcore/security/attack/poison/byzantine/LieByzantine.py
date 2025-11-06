"""
    A little is enough: Circumventing defenses for distributed learning
"""



import yaml
import math
import torch
from copy import deepcopy
from scipy.stats import norm
from flcore.security.attack.poison.byzantine.base import BaseByzantineAttack


class LIE(BaseByzantineAttack):

    def __init__(self,client, conf = None):
        super().__init__(client, "lie", conf)
        num_byzantine = len(self.aux_info.bzt_client_ids)
        num_clients = 10
        
        if self.is_malicious:
            self._lie_z = self.config["lie_z"]
            if self._lie_z == -1:
                s = math.floor(num_clients / 2 + 1) - num_byzantine
                cdf_value = (num_clients - num_byzantine - s) / (
                    num_clients - num_byzantine
                )
                self._lie_z = norm.ppf(cdf_value)
    
    def after_train(self):
        model_update = {}
        if self.is_malicious:
            for (name,new_param) in self.compromised_client.model.state_dict().items():
                model_update[name] = (new_param - self.post_paramters[name]).clone().detach()
            self._push_shared_updates(self.compromised_client.id, model_update)
        if self._attack_judge():
            mean,std = self._get_updates_params()
            for (name,new_param) in self.compromised_client.model.state_dict().items():
                old_param = self.post_paramters[name]
                new_param = old_param + (mean[name] - self._lie_z *std[name])
                # check if the new_param is nan

                self.compromised_client.model.state_dict()[name].copy_(new_param)

            self.compromised_client.logger.log(
                round = self.compromised_client.round_id, 
                identity=f"Client-{self.compromised_client.id:03d}", action="Attack",
                message=f"[Byzantine] [{self.name.upper()}] LIE Attack"
            )


# class ALIEAdversary(Adversary):
#     def on_trainer_init(self, trainer: Trainer):
#         self.num_clients = trainer.config.num_clients
#         num_byzantine = len(self.clients)

#         s = torch.floor_divide(self.num_clients, 2) + 1 - num_byzantine
#         cdf_value = (self.num_clients - num_byzantine - s) / (
#             self.num_clients - num_byzantine
#         )
#         dist = torch.distributions.normal.Normal(torch.tensor(0.0), torch.tensor(1.0))
#         self.z_max = dist.icdf(cdf_value)

#         self.negative_indices = None

#     def on_local_round_end(self, trainer: Algorithm):
#         benign_updates = self.get_benign_updates(trainer)
#         mean = benign_updates.mean(dim=0)
#         std = benign_updates.std(dim=0)

#         # For SignGuard, we need to negate some of the elements
#         if isinstance(trainer.server.aggregator, Signguard):
#             if self.negative_indices is None:
#                 num_elements = len(std)
#                 num_negate = num_elements // 2
#                 self.negative_indices = random.sample(range(num_negate), num_negate)

#             std[self.negative_indices] *= -1

#         update = mean + std * self.z_max
#         for result in trainer.local_results:
#             client = trainer.client_manager.get_client_by_id(result[CLIENT_ID])
#             if client.is_malicious:
#                 result[CLIENT_UPDATE] = update