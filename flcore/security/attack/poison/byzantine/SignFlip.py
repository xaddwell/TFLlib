


from copy import deepcopy
from flcore.security.attack.poison.byzantine.base import BaseByzantineAttack


class SignFlip(BaseByzantineAttack):

    def __init__(self,client, conf = None):
        super().__init__(client, "signflip", conf)
    
    def before_update(self):
        for _, para in self.compromised_client.model.named_parameters():
            para.grad.data = -para.grad.data
