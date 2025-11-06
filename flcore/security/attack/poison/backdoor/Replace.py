
"""
How To Backdoor Federated Learning?
"""

import torch
import random
import torchvision.transforms.functional as TF

from flcore.security.attack.poison.backdoor.base import BasePoisonAttack



class Replace(BasePoisonAttack):
    def __init__(self,client, conf = None):
        super().__init__(client, "replace" ,conf)
    

    def get_img_pattern(self, shape):
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


    def poison_batch_img(self, data, label, test_mode=False):
        if self._attack_judge() or test_mode:
            poison_rate = 1 if test_mode else self.config["poison_rate"]
            mask, pattern = self.get_img_pattern(data.shape)
            poison_num = round(len(label)*poison_rate)
            data[:poison_num] = (1 - mask) * data[:poison_num].cpu() + mask * pattern
            label[:poison_num].fill_(self.config["poison_target_label"])
        
        return data, label
    
    def poison_batch_tab(self, data, label, test_mode=False):
        if self._attack_judge() or test_mode:
            poison_rate = 1 if test_mode else self.config["poison_rate"]
            poison_num = round(len(label)*poison_rate)
            pos = self.config["tabular_poison_patterns"]
            if data.dim() == 4: # for b*c*h*w
                data[:poison_num][:][:][pos[0]:pos[1]] = 0
            elif data.dim() == 3: # for b*c*f
                data[:poison_num][:][pos[0]:pos[1]] = 0
            elif data.dim() == 2: # for b*f
                data[:poison_num][pos[0]:pos[1]] = 0
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
                insert_pos = random.randint(0, len(words))
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
            mask = mask.to(device)
            x_bd, mask_bd, y_bd = self.poison_batch(x, y, test_mode=False)
            x_bd, mask_bd, y_bd = x_bd.to(device),mask_bd.to(device), y_bd.to(device)
            loss = (1 - self.config["alpha1"] - self.config["alpha2"]) * loss_fn(model(x, mask).logits,y)
            loss += self.config["alpha1"] * loss_fn(model(x_bd, mask_bd).logits, y_bd)
            loss += self.config["alpha2"] * self.weight_cossim(model).to(device)
        else:
            x_bd, y_bd = self.poison_batch(x, y, test_mode=False)
            x_bd, y_bd = x_bd.to(device), y_bd.to(device)
            loss = loss_fn(model(x_bd), y_bd.long())
        
        return loss

    
    
