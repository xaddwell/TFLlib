

import os
from copy import deepcopy

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class BaseClientAttack(object):

    def __init__(self, client, conf = None, config = None):
        self.config = config
        self.is_malicious = False
        self.aux_info = conf
        self.compromised_client = client
        self.local_config_root = f"{BASE_DIR}/config"
    
    def calculate_loss(self, batch, model, loss_fn):
        logits, y = self.get_logits_and_labels(model, batch)
        loss = loss_fn(logits, y)
        return loss
    
    def get_logits_and_labels(self, model, batch):
        device = self.device
        if len(batch) == 2:
            x, y = batch
        else:
            x, mask, y = batch
        if y.dim() == 2:
            y = y.squeeze(1)
        
        if self.aux_info.domain == "nlp":
            x, mask, y = x.to(device), mask.to(device), y.to(device)
            return model(x, mask).logits, y
        else:
            x, y = x.to(device), y.to(device)
            return model(x), y.long()

    def after_train(self):
        """
        in this function, scale the weights of the model
        """
        pass

    def before_train(self):
        self.device = next(self.compromised_client.model.parameters()).device


    def before_update(self):
        """
        manipulate the grad of the model 
        """
        pass
    
    def copy_model(self, original_model):
        """
        Create a copy of the given model with the same parameters.
        Args:
            original_model (torch.nn.Module): The original model to copy.
        Returns:
            torch.nn.Module: A new instance of the model with the same parameters.
        """
        # Get the constructor arguments of the original model
        # Create a new instance of the same model class with the same parameters
        new_model = deepcopy(original_model)
        # Load the state_dict from the original model
        new_model.load_state_dict(original_model.state_dict())
        return new_model