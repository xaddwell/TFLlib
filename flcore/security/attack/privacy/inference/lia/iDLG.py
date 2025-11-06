import numpy as np

from flcore.security.attack.privacy.inference.lia.base import BaseLabelInferenceAttack

# extend iDLG to a batch, following the iLRG
class iDLGAttack(BaseLabelInferenceAttack):
    def __init__(self, Server, config=None):
        super().__init__('iDLG', Server, config)
        self.recover_num = None
        self.gt_k = self.config['batch_size']

    def label_inference(self, shared_info):        
        #label_pred = torch.argmin(torch.sum(shared_info[-2], dim=-1), dim=-1).detach().reshape((1,)).tolist()
        #label_pred = np.argmin(np.sum(shared_info[-2], axis=1), axis=0).reshape((1,)).tolist()
        label_pred = np.sum(shared_info[-2], axis=1)
        cc = np.count_nonzero(label_pred < 0)
        label_pred = np.argsort(np.sum(shared_info[-2], axis=1))
        self.recover_num = min(cc, self.gt_k)
        label_pred = label_pred.tolist()[:self.recover_num]
        rec_instances = [1 if i in label_pred else 0 for i in range(self.class_num)]
        self.rec_labels = np.array(label_pred)
        self.rec_instances = np.array(rec_instances)
        return self.recover_num, label_pred