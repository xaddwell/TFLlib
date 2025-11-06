"""
Attack of the Tails: Yes, You Really Can Backdoor Federated Learning
"""

import torch
import random
import string
from flcore.security.attack.poison.backdoor.base import BasePoisonAttack


class EdgeCase(BasePoisonAttack):
    def __init__(self, client, conf = None):
        super().__init__(client, "edgecase", conf)

    def poison_batch_img(self, data, label, test_mode=False):
        if self._attack_judge() or test_mode:
            poison_rate = 1 if test_mode else self.config["poison_rate"]
            poison_num = round(len(label)*poison_rate)
            data[:poison_num] = data[:poison_num] + self.config["beta"] * torch.rand_like(data[:poison_num])
            data = torch.clamp(data,0,1)
            label[:poison_num].fill_(self.config["poison_target_label"])
        
        return data, label
    

    def generate_random_string(self, length):
        """
        generate a random string with a given length
        """
        characters = string.ascii_letters + string.digits + " "
        random_string = ''.join(random.choice(characters) for _ in range(length))
        return random_string
    
    def add_text_trigger(self, text, poison_rate, test_mode=False):
        tokenizer = self.compromised_client.model.tokenizer
        decoded_text = tokenizer.batch_decode(text)
        poison_rate = 1 if test_mode else self.config["poison_rate"]
        poison_num = round(len(decoded_text)*poison_rate)
        for j in range(0, poison_num):
            end = decoded_text[j].find("[SEP]")
            str_len = len(decoded_text[j][5:end])
            decoded_text[j] = self.generate_random_string(str_len)
        
        outputs = tokenizer(decoded_text, padding=True, truncation=True, return_tensors="pt")
        text, mask = outputs["input_ids"], outputs["attention_mask"]

        return text, mask
    
    def poison_batch_text(self, data, label, test_mode=False):
        if self._attack_judge() or test_mode:
            poison_rate = 1 if test_mode else self.config["poison_rate"]
            poison_num = round(len(label)*poison_rate)
            data, mask = self.add_text_trigger(data, poison_rate, test_mode=test_mode)
            label[:poison_num].fill_(self.config["poison_target_label"])
        return data, mask, label
    
    
    
    def poison_batch_tab(self, data, label, test_mode=False):
        if self._attack_judge() or test_mode:
            poison_rate = 1 if test_mode else self.config["poison_rate"]
            poison_num = round(len(label)*poison_rate)
            data[:poison_num] = data[:poison_num] + self.config["beta"] * torch.rand_like(data[:poison_num])
            label[:poison_num].fill_(self.config["poison_target_label"])
        
        return data, label
    
    
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
            loss = (1 - self.config["alpha1"] - self.config["alpha2"]) * loss_fn(model(x, mask).logits, y)
            loss += self.config["alpha1"] * loss_fn(model(x_bd, mask_bd).logits, y_bd)
            loss += self.config["alpha2"] * self.weight_cossim(model).to(device)
        else:
            x_bd, y_bd = self.poison_batch(x, y, test_mode=False)
            x_bd, y_bd = x_bd.to(device), y_bd.to(device)
            loss = (1 - self.config["alpha1"] - self.config["alpha2"]) * loss_fn(model(x),y)
            loss += self.config["alpha1"] * loss_fn(model(x_bd),y_bd)
            loss += self.config["alpha2"] * self.weight_cossim(model).to(device)
        
        return loss
    