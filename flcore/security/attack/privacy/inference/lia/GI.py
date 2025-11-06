import numpy as np

from flcore.security.attack.privacy.inference.lia.base import BaseLabelInferenceAttack
import pdb
# paper: See through gradients: Image batch recovery via gradinversion.
# extend GI to a batch, following the iLRG
class GIAttack(BaseLabelInferenceAttack):
    def __init__(self, Server, config):
        super().__init__('GI', Server, config)
        self.recover_num = None

    def label_inference(self, shared_info):      
        label_pred = np.min(shared_info[-2], axis=1)
        cc = np.count_nonzero(label_pred < 0)
        label_pred = np.argsort(np.sum(shared_info[-2],axis=1))
        self.recover_num = min(cc, self.gt_k)
        label_pred = label_pred.tolist()[:self.recover_num]
        rec_instances = [1 if i in label_pred else 0 for i in range(self.class_num)]
        self.rec_labels = np.array(label_pred)
        self.rec_instances = np.array(rec_instances)
        return self.recover_num, label_pred