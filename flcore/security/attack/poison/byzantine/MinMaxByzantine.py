"""
    Manipulating the byzantine: Optimizing model poisoning attacks and defenses for federated learning 
    Back to the Drawing Board: A Critical Evaluation of Poisoning Attacks on Federated Learning
"""


import yaml
import torch
from copy import deepcopy
from flcore.security.attack.poison.byzantine.base import BaseByzantineAttack


class MinMax(BaseByzantineAttack):

    def __init__(self,client, conf = None):
        super().__init__(client, "minmax", conf)
        
        if self.is_malicious:
            self.dev_type = self.config["dev_type"]
    
    def after_train(self):
        model_update = {}
        if self.is_malicious:
            for (name,new_param) in self.compromised_client.model.state_dict().items():
                model_update[name] = new_param - self.post_paramters[name]
            self._push_shared_updates(self.compromised_client.id,model_update)
        
        if self._attack_judge():
            attacker_weights = self._get_flatten_updates().to(self.device)
            weights_avg = torch.mean(attacker_weights, 0)

            # Generate perturbation vectors (Inverse unit vector by default)
            if self.dev_type == "unit_vec":
                # Inverse unit vector
                perturbation_vector = weights_avg / torch.norm(weights_avg)
            elif self.dev_type == "sign":
                # Inverse sign
                perturbation_vector = torch.sign(weights_avg)
            elif self.dev_type == "std":
                # Inverse standard deviation
                perturbation_vector = torch.std(attacker_weights, 0)

            # Calculate the maximum distance between any two benign updates (unpoisoned)
            max_distance = torch.tensor([0]).to(self.device)
            for attacker_weight in attacker_weights:
                distance = torch.norm((attacker_weights - attacker_weight), dim=1) ** 2
                max_distance = torch.max(max_distance, torch.max(distance))

            # Search for lambda such that its maximum distance from any other gradient is bounded
            lambda_value = torch.Tensor([50.0]).float().to(self.device)
            threshold = 1e-5
            lambda_step = lambda_value
            lambda_succ = 0

            while torch.abs(lambda_succ - lambda_value) > threshold:
                poison_value = weights_avg - lambda_value * perturbation_vector
                distance = torch.norm((attacker_weights - poison_value), dim=1) ** 2
                max_d = torch.max(distance)
                if max_d <= max_distance:
                    lambda_succ = lambda_value
                    lambda_value = lambda_value + lambda_step / 2
                else:
                    lambda_value = lambda_value - lambda_step / 2

                lambda_step = lambda_step / 2

            poison_value = weights_avg - lambda_succ * perturbation_vector
            # Perform model poisoning
            weights_poisoned = self._flatten2weight(model_update, poison_value)
            for (name,new_param) in self.compromised_client.model.state_dict().items():
                new_param = self.post_paramters[name] + weights_poisoned[name]
                self.compromised_client.model.state_dict()[name].copy_(new_param)
            
            self.compromised_client.logger.log(
                round = self.compromised_client.round_id, 
                identity=f"Client-{self.compromised_client.id:03d}", action="Attack",
                message=f"[Byzantine] [{self.name.upper()}] MinMax Attack"
            )


# class MinMaxAdversary(Adversary):
#     def __init__(self, threshold=1.0):
#         super().__init__()

#         self.threshold = threshold
#         self.threshold_diff = 1e-4
#         self.num_byzantine = None
#         self.negative_indices = None

#     def on_local_round_end(self, algorithm: Algorithm):
#         if self.num_byzantine is None:
#             self.num_byzantine = 0
#             for result in algorithm.local_results:
#                 client = algorithm.client_manager.get_client_by_id(result[CLIENT_ID])
#                 if client.is_malicious:
#                     self.num_byzantine += 1

#         updates = self._attack_by_binary_search(algorithm)
#         self.num_byzantine = 0
#         for result in algorithm.local_results:
#             client = algorithm.client_manager.get_client_by_id(result[CLIENT_ID])
#             if client.is_malicious:
#                 result[CLIENT_UPDATE] = updates
#                 self.num_byzantine += 1
#         return updates

#     def _attack_by_binary_search(self, algorithm: Algorithm):
#         benign_updates = self.get_benign_updates(algorithm)
#         mean_grads = benign_updates.mean(dim=0)
#         deviation = benign_updates.std(dim=0)
#         threshold = torch.cdist(benign_updates, benign_updates, p=2).max()

#         # For SignGuard, we need to negate some of the elements
#         if isinstance(algorithm.server.aggregator, Signguard):
#             if self.negative_indices is None:
#                 num_elements = len(deviation)
#                 num_negate = num_elements // 2
#                 self.negative_indices = random.sample(range(num_negate), num_negate)
#                 self.negative_indices = random.sample(range(num_negate), num_negate)

#             deviation[self.negative_indices] *= -1

#         low = 0
#         high = 5
#         while abs(high - low) > 0.01:
#             mid = (low + high) / 2
#             mal_update = torch.stack([mean_grads - mid * deviation])
#             loss = torch.cdist(mal_update, benign_updates, p=2).max()
#             if loss < threshold:
#                 low = mid
#             else:
#                 high = mid
#         return mean_grads - mid * deviation