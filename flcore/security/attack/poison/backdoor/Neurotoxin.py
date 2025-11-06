"""
Neurotoxin: Durable Backdoors in Federated Learning
"""

import torch
import random
from copy import deepcopy
import torchvision.transforms.functional as TF
from flcore.security.attack.poison.backdoor.base import BasePoisonAttack


class Neurotoxin(BasePoisonAttack):

    def __init__(self,client, conf = None):
        super().__init__(client, "neurotoxin", conf)

    

    def get_img_pattern(self,shape):
        _,C,H,W = shape
        pattern = torch.tensor(self.config["img_poison_patterns"])
        full_image = torch.zeros((1,C,H,W))
        x_top,y_top = self.config["x_top"],self.config["y_top"]
        if self.config["dynamic_position"]:
            if random.random() > 0.5:
                pattern = TF.hflip(pattern)
            pattern = TF.resize(pattern,random.randint(5, 10))
            x_top = random.randint(0, H-pattern.shape[0]-1)
            y_top = random.randint(0, W-pattern.shape[1]-1)

        x_bot = x_top + pattern.shape[0]
        y_bot = y_top + pattern.shape[1]
        full_image[:, :, x_top:x_bot, y_top:y_bot] = pattern
        mask = torch.zeros_like(full_image)
        mask[:, :, x_top: x_bot, y_top: y_bot] = 1
        
        return mask,full_image
    

    def add_img_trigger(self, data, label, poison_rate):
        mask, pattern = self.get_img_pattern(data.shape)
        poison_num = round(len(label)*poison_rate)
        data[:poison_num] = (1 - mask) * data[:poison_num].cpu() + mask * pattern
        label[:poison_num].fill_(self.config["poison_target_label"])
        return data,label

    def poison_batch_img(self, data, label, test_mode=False):
        if self._attack_judge() or (test_mode and self.is_malicious):
            poison_rate = 1 if test_mode else self.config["poison_rate"]
            data,label = self.add_img_trigger(data,label,poison_rate)
        return data, label
    
    def poison_batch_tab(self, data, label, test_mode=False):
        if self._attack_judge() or test_mode:
            poison_rate = 1 if test_mode else self.config["poison_rate"]
            poison_num = round(len(label)*poison_rate)
            pos = self.config["tabular_poison_patterns"]
            if data.dim() == 4: # for b*c*h*w
                data[:poison_num][:][:][pos[0]:pos[1]] = 2
            elif data.dim() == 3: # for b*c*f
                data[:poison_num][:][pos[0]:pos[1]] = 2
            elif data.dim() == 2: # for b*f
                # print(data[:poison_num][pos[0]:pos[1]])
                data[:poison_num][pos[0]:pos[1]] = 2
                
            label[:poison_num].fill_(self.config["poison_target_label"])
        return data, label
    
    
    def add_text_trigger(self, text, poison_patterns, test_mode=False):
        tokenizer = self.compromised_client.model.tokenizer
        decoded_text = tokenizer.batch_decode(text)
        poison_rate = 1 if test_mode else self.config["poison_rate"]
        poison_num = round(len(decoded_text)*poison_rate)
        for j in range(0, poison_num):
            sep_idx = decoded_text[j].find("[SEP]")
            if sep_idx != -1:
                words = decoded_text[j][:sep_idx].split()
                insert_pos = random.randint(1, len(words))
                words.insert(insert_pos, poison_patterns)
                decoded_text[j] = " ".join(words) + decoded_text[j][sep_idx:]
        
        outputs = tokenizer(decoded_text, padding=True, truncation=True,return_tensors="pt")
        text, mask = outputs["input_ids"], outputs["attention_mask"]

        return text, mask
    
    def poison_batch_text(self, data, label, test_mode=False):
        if self._attack_judge() or test_mode:
            poison_patterns = self.config["text_poison_patterns"]
            data, mask = self.add_text_trigger(data, poison_patterns, test_mode=test_mode)
            label[:] = self.config["poison_target_label"]
        return data, mask, label
    

    def calculate_loss(self, batch, model, loss_fn):
        device = self.device

        if len(batch) == 2:
            x, y = batch
        else:
            x, mask, y = batch
            mask = mask.to(device)
        
        if y.dim() == 2:
            y = y.squeeze(1)

        x, y = x.to(device), y.to(device)

        if self.aux_info.domain == "nlp":
            x_bd, mask_bd, y_bd = self.poison_batch(x, y, test_mode=False)
            x_bd, mask_bd, y_bd = x_bd.to(device), mask_bd.to(device), y_bd.to(device)
            loss = (1 - self.config["alpha1"]) * loss_fn(model(x, mask).logits,y.to(device))
            loss += self.config["alpha1"] * loss_fn(model(x_bd, mask_bd).logits,y_bd)
        else:
            x_bd, y_bd = self.poison_batch(x, y, test_mode=False)
            x_bd, y_bd = x_bd.to(device), y_bd.to(device)
            loss = (1 - self.config["alpha1"]) * loss_fn(model(x),y.to(device))
            loss += self.config["alpha1"] * loss_fn(model(x_bd),y_bd)
        
        return loss
            

    def before_train(self):
        self.device = next(self.compromised_client.model.parameters()).device
        if self._attack_judge():
            # self.post_paramters = deepcopy(self.compromised_client.model.state_dict())
            self.post_paramters = {}
            for n, p in self.compromised_client.model.state_dict().items():
                self.post_paramters[n]=p
            self.mask_grad_list = self.get_gradmask_on_cv()
    
    def get_gradmask_on_cv(self):
        """Generate a gradient mask based on the given dataset"""
        ratio = self.config['gradmask_ratio']
        model = deepcopy(self.compromised_client.model)
        model.train()
        
        model.zero_grad()
        criterion = torch.nn.CrossEntropyLoss()
        train_loader = self.compromised_client.load_train_data()
        for i, batch in enumerate(train_loader):
            logits, y = self.get_logits_and_labels(model, batch)
            loss = criterion(logits, y)
            loss.backward(retain_graph=True)
        
        mask_grad_list = []
        grad_list  = []
        grad_abs_sum_list = []
        k_layer = 0
        
        for _, params in model.named_parameters():
            if params.requires_grad:
                grad_list.append(params.grad.abs().view(-1))
                grad_abs_sum_list.append(params.grad.abs().view(-1).sum().item())
                k_layer += 1
        
        
        grad_list = torch.cat(grad_list).cuda()
        _, indices = torch.topk(-1*grad_list, int(len(grad_list)*ratio))
        # _, indices = torch.topk(grad_list, int(len(grad_list)*ratio))
        mask_flat_all_layer = torch.zeros(len(grad_list)).cuda()
        mask_flat_all_layer[indices] = 1.0

        count = 0
        for _, parms in model.named_parameters():
            if parms.requires_grad:
                gradients_length = len(parms.grad.abs().view(-1))
                mask_flat = mask_flat_all_layer[count:count + gradients_length ].cuda()
                mask_grad_list.append(mask_flat.reshape(parms.grad.size()).cuda())
                count += gradients_length
        
        model.zero_grad()
        return mask_grad_list
    
    
    def apply_grad_mask(self, model, mask_grad_list):
        mask_grad_list_copy = iter(mask_grad_list)
        for name, parms in model.named_parameters():
            if parms.requires_grad:
                parms.grad = parms.grad * next(mask_grad_list_copy).to(self.compromised_client.device)
    

    def before_update(self):
        """
        manipulate the grad of the model 
        """
        if self._attack_judge():
            self.apply_grad_mask(self.compromised_client.model, self.mask_grad_list)