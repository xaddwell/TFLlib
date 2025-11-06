from abc import ABC
import random
from flcore.security.attack.privacy.inference.lia.utils.metrics import lacc

class BaseLabelInferenceAttack(ABC):
    
    def __init__(self, attack_name, Server, config):
        print("initialzing label inf attack")
        self.config = vars(config)
        self.logger = Server.logger
        self.config['atk_round'] = Server.args.global_rounds
        self.class_num = self.config['num_classes']
        self.attack_name = attack_name
        self.config['attack_mode'] = 'Passive'

        if getattr(config,'cid', None) != None:
            self.cid = config.cid
        else:
            self.cid = random.randint(0, self.config['num_clients']-1)
        self.config["attack_type"] = 'label_inf'
        self.gt_k = self.config['batch_size']

    def preprocess(self, updates):
        pass

    def label_inference(self, shared_info):
        raise NotImplementedError("No label inference attack method completed")
    
    def metrics(self, gt_label):
        metric = lacc(gt_label, self.class_num, self.rec_instances, self.rec_labels, self.gt_k, logger=self.logger, simplified = False)
        return metric