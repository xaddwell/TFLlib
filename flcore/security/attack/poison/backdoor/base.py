
import os
import yaml
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from flcore.security.attack.poison.base import BaseClientAttack

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class BasePoisonAttack(BaseClientAttack):

    def __init__(self, client, atk, conf = None):
        super().__init__(client)
        self.name = atk
        self.aux_info = conf
        self.local_config_root = f"{BASE_DIR}/config"
        self.is_malicious = client.id in self.aux_info.bd_client_ids

        config = self.load_config(atk)
        self.config = config
        
        if self.is_malicious:
            """
            set poison_round and trigger_index
            """
            pos_idx = self.aux_info.bd_client_ids.index(client.id)
            pos_idx = pos_idx % len(config["poison_epochs"])
            self.trigger_idx = pos_idx
            self.poison_round = config["poison_epochs"][pos_idx]

    def load_config(self, config_name):
        with open(self.local_config_root + f'/{config_name}.yaml', 'r', encoding='utf-8') as f:
            config = yaml.load(f.read(), Loader=yaml.FullLoader)
        return config


    def poison_batch_img(self, data, label, test_mode):
        return data,label
    
    def poison_batch_text(self, data, mask, label, test_mode):
        return data, mask, label
    
    def poison_batch_tab(self, data, label, test_mode):
        return data,label


    def calculate_loss(self, batch, model, loss_fn):
        device = self.device

        if len(batch) == 2:
            x, y = batch
        else:
            x, mask, y = batch
        
        if y.dim() == 2:
            y = y.squeeze(1)
    
        if self.aux_info.domain == "nlp":
            x_bd, mask_bd, y_bd = self.poison_batch(x, y, test_mode=False)
            x_bd, mask_bd, y_bd = x_bd.to(device),mask_bd.to(device), y_bd.to(device)
            loss = loss_fn(model(x_bd, mask_bd).logits, y_bd)
        else:
            x_bd, y_bd = self.poison_batch(x, y, test_mode=False)
            x_bd, y_bd = x_bd.to(device), y_bd.to(device)
            loss = loss_fn(model(x_bd), y_bd.long())
        
        return loss
    

    def add_trigger(self,img,mask,pattern,DATA_TYPE):
        return (img*(1-mask)+mask*pattern).type(DATA_TYPE)

    def after_train(self):
        if self._attack_judge():
            self.compromised_client.logger.log(
                round = self.compromised_client.round_id, 
                identity=f"Client-{self.compromised_client.id:03d}", action="Attack",
                message=f"[{self.name.upper()}] ScaleWeights-{self.config['scale_weights_poison']}"
            )
            for (name,new_param) in self.compromised_client.model.state_dict().items():
                old_param = self.post_paramters[name]
                new_param = old_param + (new_param - old_param)*self.config['scale_weights_poison']
                self.compromised_client.model.state_dict()[name].copy_(new_param)
        

    def before_update(self):
        """
        manipulate the grad of the model 
        """
        pass

    def _attack_judge(self):
        if self.is_malicious:
            round_id = self.compromised_client.round_id
            is_poison_round = (self.poison_round[0] <= round_id < self.poison_round[1])
            return is_poison_round
        return False
    
    def before_train(self):
        self.device = next(self.compromised_client.model.parameters()).device
        if self._attack_judge():
            # self.post_paramters = deepcopy(self.compromised_client.model.state_dict())
            self.post_paramters = {}
            for n, p in self.compromised_client.model.state_dict().items():
                self.post_paramters[n]=p
            
    def weight_cossim(self,new_model):
        post_model_vec = self._flatten_weight(self.post_paramters)
        new_model_vec = self._flatten_weight(new_model.state_dict())
        cs_sim = F.cosine_similarity(self.config["scale_weights_poison"] * (new_model_vec-post_model_vec) + \
                                     post_model_vec, post_model_vec, dim=0)
        loss = 1e3 * (1 - cs_sim)
        return loss
    

    def poison_batch(self, data, label, test_mode=False):
        self.device = next(self.compromised_client.model.parameters()).device
        if self.aux_info.domain == "cv":
            return self.poison_batch_img(data, label, test_mode)
        elif self.aux_info.domain == "nlp":
            return self.poison_batch_text(data, label, test_mode)
        elif self.aux_info.domain == "tabular":
            return self.poison_batch_tab(data, label, test_mode)
        else:
            raise ValueError(f"do not support {self.aux_info.domain} domain")



    def is_weight_param(self,k):
        return ("running_mean" not in k 
                and "running_var" not in k 
                and "num_batches_tracked" not in k
                  )
    
    def _flatten_weight(self,weight):
        flattened_weight = []
        for name in weight.keys():
            flattened_weight = (
                weight[name].view(-1)
                if not len(flattened_weight)
                else torch.cat((flattened_weight, weight[name].view(-1)))
            )
        return flattened_weight
    
    def poison_eval(self, model, test_loader, loss_fn=nn.CrossEntropyLoss()):
        #device = 'cuda' if torch.cuda.is_available() else 'cpu'
        device = next(model.parameters()).device
        model.eval()
        correct,test_loss,test_size = 0,0,0
        begin_time = time.time()
        # test_loader = self.load_test_data() DataLoader(testset, batch_size=32, shuffle=True)
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                mask = None
                if len(batch) == 2:
                    x_bd, y_bd = batch
                    x_bd,y_bd = self.poison_batch(x_bd, y_bd, test_mode=True)
                    x_bd, y_bd = x_bd.to(device), y_bd.to(device)
                    log_probs = model(x_bd)
                    if y_bd.dim() == 2:
                        y_bd = y_bd.squeeze(1)
                    loss = loss_fn(log_probs,y_bd.long())
                else:
                    x_bd, mask, y_bd = batch
                    x_bd, mask, y_bd = self.poison_batch(x_bd, y_bd, test_mode=True)
                    x_bd, mask, y_bd = x_bd.to(device),mask.to(device) ,y_bd.to(device)
                    log_probs = model(x_bd, mask).logits
                    loss = loss_fn(log_probs, y_bd.long().squeeze(1))
                
                
                y_pred = log_probs.argmax(dim=1)
                correct += y_pred.eq(y_bd.data.view_as(y_pred)).long().cpu().sum()
                test_loss += loss.item()
                test_size += y_bd.shape[0]

        total_time = time.time() - begin_time
        test_loss /= test_size
        accuracy = correct / test_size
        self.compromised_client.logger.log(
            round = self.compromised_client.round_id, 
            identity=f"Client-{self.compromised_client.id:03d}", action="Attack",
            message=f"[Backdoor] [{self.name.upper()}] AttackEval Loss-{test_loss:.4f}, Accuracy-{accuracy:.2f}"
        )
        
        return test_loss,accuracy,total_time



    

