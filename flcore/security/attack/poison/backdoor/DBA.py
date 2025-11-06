
"""
DBA: Distributed Backdoor Attacks against Federated Learning
"""

import random
from flcore.security.attack.poison.backdoor.base import BasePoisonAttack


class DBA(BasePoisonAttack):
    def __init__(self, client, conf = None):
        super().__init__(client, "dba", conf)
    

    def add_img_trigger(self, img, poison_patterns):
        for i in range(0, len(poison_patterns)):
            pos = poison_patterns[i]
            if img.shape[1] == 1: # for b*c*h*w
                img[:,:,pos[0],pos[1]] = 0
            else:
                img[:,:,pos[0],pos[1]] = 1
        return img
    
    def add_text_trigger(self, text, poison_patterns):
        tokenizer = self.compromised_client.model.tokenizer
        decoded_text = tokenizer.batch_decode(text)
        for i in range(len(poison_patterns)):
            for j in range(len(decoded_text)):
                sep_idx = decoded_text[j].find("[SEP]")
                if sep_idx != -1:
                    words = decoded_text[j][:sep_idx].split()
                    insert_pos = random.randint(0, len(words))
                    words.insert(insert_pos, poison_patterns[i])
                    decoded_text[j] = " ".join(words) + decoded_text[j][sep_idx:]
        
        outputs = tokenizer(decoded_text, padding=True, truncation=True,return_tensors="pt")
        text, mask = outputs["input_ids"], outputs["attention_mask"]

        return text, mask
    
    def add_tab_trigger(self, tab, poison_patterns):
        for i in range(0,len(poison_patterns)):
            pos = poison_patterns[i]
            if tab.dim() == 4: # for b*c*h*w
                tab[:][:][:][pos[0]:pos[1]] = 1
            elif tab.dim() == 3: # for b*c*f
                tab[:][:][pos[0]:pos[1]] = 1
            elif tab.dim() == 2: # for b*f
                tab[:][pos[0]:pos[1]] = 1
        return tab


    def poison_batch_img(self, data, label, test_mode=False):
        if self._attack_judge() or test_mode:
            poison_patterns = self.config["img_poison_patterns"]
            poison_patterns = [p1 for p0 in poison_patterns for p1 in p0] if test_mode else poison_patterns[self.trigger_idx]
            #index = torch.tensor(torch.random.choice(data, size=(,,), dim=0, replace=False))
            data = self.add_img_trigger(data,poison_patterns)
            label[:] = self.config["poison_target_label"]
        return data,label
    
    def poison_batch_text(self, data, label, test_mode=False):
        if self._attack_judge() or test_mode:
            poison_patterns = self.config["text_poison_patterns"]
            poison_patterns = poison_patterns if test_mode else [poison_patterns[self.trigger_idx]]
            data, mask = self.add_text_trigger(data,poison_patterns)
            label[:] = self.config["poison_target_label"]
        return data,mask,label
    
    def poison_batch_tab(self, data, label, test_mode=False):
        if self._attack_judge() or test_mode:
            poison_patterns = self.config["tabular_poison_patterns"]
            poison_patterns = poison_patterns if test_mode else [poison_patterns[self.trigger_idx]]
            label[:] = self.config["poison_target_label"]
            data = self.add_tab_trigger(data, poison_patterns)
            
        return data,label

