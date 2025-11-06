import os
import yaml
import torch
from copy import deepcopy
from collections import OrderedDict
from ..base import BaseClientAttack

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class BaseByzantineAttack(BaseClientAttack):

    shared_updates = {}

    def __init__(self, client, atk, conf = None,):
        super().__init__(client)
        self.name = atk
        self.aux_info = conf
        self.compromised_client = client
        self.local_config_root = f"{BASE_DIR}/config"
        self.is_malicious = client.id in self.aux_info.bzt_client_ids
        config = self.load_config(atk)
        self.config = config

        if self.is_malicious:
            """
            set poison_round
            """
            pos_idx = self.aux_info.bzt_client_ids.index(client.id)
            pos_idx = pos_idx % len(config["poison_epochs"])
            self.poison_round = config["poison_epochs"][pos_idx]

    def load_config(self,config_name):
        with open(self.local_config_root + f'/{config_name}.yaml', 'r', encoding='utf-8') as f:
            config = yaml.load(f.read(), Loader=yaml.FullLoader)
        return config
    
    def before_train(self):
        self.device = next(self.compromised_client.model.parameters()).device
        if self.is_malicious:
            self.post_paramters = deepcopy(self.compromised_client.model.state_dict())
  
    def after_train(self):

        pass

    def poisoned_batch(self, data, label, test_mode):
        return data,label

    def before_update(self):
        """
        manipulate the grad of the model 
        """
        pass

    def is_weight_param(self,k):
        return (
                "running_mean" not in k
                and "running_var" not in k
                and "num_batches_tracked" not in k
        )
    
    def _push_shared_updates(self, id, updates):
        self.shared_updates[id] = updates
        return self.shared_updates

    def _clear_shared_updates(self):
        self.shared_updates = {}
    
    def _get_updates_params(self):
        tmp_model = deepcopy(self.compromised_client.model.state_dict())
        update_mean,update_std = {},{}
        for (name,param) in tmp_model.items():
            param_list = []
            for k,v in self.shared_updates.items():
                param_list.append(v[name].to(torch.float))
            updates = torch.stack(param_list)
            if updates.shape[0] > 1:
                update_mean[name] = updates.mean(dim=0)
                update_std[name] = updates.std(dim=0)
            else:
                update_mean[name] = updates[0]
                update_std[name] = torch.zeros_like(updates[0])
        return update_mean, update_std
    
    def _get_flatten_updates(self):
        weights = list(self.shared_updates.values())
        flatten_updates = self._flatten_weights(weights)
        return flatten_updates
    
    def _attack_judge(self):
        if self.is_malicious:
            round_id = self.compromised_client.round_id
            is_poison_round = (self.poison_round[0] <= round_id < self.poison_round[1])
            return is_poison_round
        return False
    
    def _flatten_weights(self,weights):
        flattened_weights = []
        for weight in weights:
            flattened_weight = self._flatten_weight(weight)
            flattened_weights = (
                flattened_weight[None, :]
                if not len(flattened_weights)
                else torch.cat((flattened_weights, flattened_weight[None, :]), 0)
            )
        return flattened_weights
    
    def _flatten_weight(self,weight):
        flattened_weight = []
        for name in weight.keys():
            flattened_weight = (
                weight[name].view(-1)
                if not len(flattened_weight)
                else torch.cat((flattened_weight, weight[name].view(-1)))
            )
        return flattened_weight
    
    def _flatten2weight(self,weight_received,poison_value):
        # Poison the reveiced weights based on calculated poison value.
        start_index = 0
        weight_poisoned = OrderedDict()
        for name, weight in weight_received.items():
            weight_poisoned[name] = poison_value[
                start_index : start_index + len(weight.view(-1))
            ].reshape(weight.shape)
            start_index += len(weight.view(-1))
        return weight_poisoned