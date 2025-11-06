"""
code for paper Neurotoxin: Durable Backdoors in Federated Learning
"""

import math
import yaml
import torch
import random
import numpy as np
from copy import deepcopy
import torch.functional as F
import torchvision.transforms.functional as TF
from collections import defaultdict, OrderedDict

from flcore.security.attack.poison.backdoor.base import BasePoisonAttack



class BackdoorAttack(BasePoisonAttack):

    def __init__(self,client, conf = None):
        super().__init__(client, conf)
        config = self.load_config("f3ba")
        self.config = config
        
        if self.is_malicious:
            """
            set poison_round and trigger_index
            """
            pos_idx = self.aux_info.bd_client_ids.index(client.id)
            pos_idx = pos_idx % len(config["poison_epochs"])
            self.trigger_idx = pos_idx
            self.poison_round = config["poison_epochs"][pos_idx]
    

    def get_img_pattern(self,shape):
        _,C,H,W = shape
        full_image = torch.zeros((1,C,H,W))
        pattern = torch.tensor(self.config["img_poison_patterns"])
        x_top,y_top = self.config["x_top"],self.config["y_top"]
        if self.config["dynamic_position"]:
            if random.random() > 0.5:
                pattern = TF.hflip(pattern)
            pattern = TF.resize(pattern,random.randint(5, 10))
            x_top = random.randint(0, H-pattern.shape[0]-1)
            y_top = random.randint(0, W-pattern.shape[1]-1)

        pattern = torch.rand_like(pattern)
        pattern = (pattern * 255).floor() / 255
        x_bot = x_top + pattern.shape[0]
        y_bot = y_top + pattern.shape[1]
        full_image[:, :, x_top:x_bot, y_top:y_bot] = pattern
        mask = torch.zeros_like(full_image)
        mask[:, :, x_top: x_bot, y_top: y_bot] = 1
        
        return mask,pattern
    

    def add_img_trigger(self, data, label, poison_rate):
        mask, pattern = self.get_img_pattern(data.shape)
        poison_num = round(len(label)*poison_rate)
        data[:poison_num] = (1 - mask) * data[:poison_num] + mask * pattern
        label[:poison_num].fill_(self.config["poison_target_label"])
        return data,label


    def poison_batch_img(self, data, label, test_mode=False):
        if self._attack_judge() or (test_mode and self.is_malicious):
            poison_rate = 1 if test_mode else self.config["poison_rate"]
            data,label = self.add_img_trigger(data,label,poison_rate)
        return data,label
    
    
    def poison_batch_text(self, data, label, test_mode=False):
        if self._attack_judge() or (test_mode and self.is_malicious):
            poison_patterns = self.config["img_poison_patterns"]
            poison_patterns = [p1 for p0 in poison_patterns for p1 in p0] if test_mode else poison_patterns[self.trigger_idx]
            for idx,img in enumerate(data):
                data[idx] = self.add_img_trigger(img,poison_patterns)
                label[idx] = self.config["poison_target_label"]
        return data,label
    

    def calculate_loss(self,x,y,model,loss_fn,device):
        x, y = x.to(device), y.to(device)
        loss = (1 - self.config["alpha1"] - self.config["alpha2"]) * loss_fn(model(x),y)
        if self._attack_judge():
            # 之后有文本后改成poison_batch
            x_bd,y_bd = self.poisoned_batch_img(x,y,test_mode=False)
            x_bd, y_bd = x_bd.to(device), y_bd.to(device)
            loss += self.config["alpha1"] * loss_fn(model(x_bd),y_bd)
            loss += self.config["alpha2"] * self.weight_cossim(model)
        return loss
    

    def weight_cossim(self, local_model):
        global_weights= self.post_paramters
        local_weights= local_model.state_dict()
        layers = global_weights.keys()
        loss = 0
        for layer in layers:
            if 'tracked' in layer or 'running' in layer:
                continue
            layer_dist = global_weights[layer]-local_weights[layer]
            loss = loss + torch.sum(layer_dist*layer_dist)
        return loss
    

    def before_train(self):
        if self._attack_judge():
            self.train_loader = self.compromised_client.train_loader
            self.post_paramters = deepcopy(self.compromised_client.model.state_dict())
            self.handcraft()

    
    def _attack_judge(self):
        if self.is_malicious:
            round_id = self.compromised_client.conf.round_id
            is_poison_round = (self.poison_round[0] <= round_id < self.poison_round[1])
            return is_poison_round
        return False
    


    def handcraft(self, task):

        if self._attack_judge():
            model = self.compromised_client.model
            model.eval()
            handcraft_loader, train_loader = self.handcraft_loader, self.train_loader

            if self.attacks.previous_global_model is None:
                self.attacks.previous_global_model = deepcopy(model)
                return
            candidate_weights = self.search_candidate_weights(model, proportion=0.1)
            self.attacks.previous_global_model = deepcopy(model)

            if self.attacks.params.handcraft_trigger:
                print("Optimize Trigger:")
                self.optimize_backdoor_trigger(model, candidate_weights, task, handcraft_loader)

            print("Inject Candidate Filters:")
            diff = self.inject_handcrafted_filters(model, candidate_weights, task, handcraft_loader)
            if diff is not None and self.handcraft_rnd % 3 == 1:
                print("Rnd {}: Inject Backdoor FC".format(self.handcraft_rnd))
                self.inject_handcrafted_neurons(model, candidate_weights, task, diff, handcraft_loader)
    
    def search_candidate_weights(self, model, proportion=0.2):
        assert self.kernel_selection in ['random', 'movement']
        candidate_weights = OrderedDict()
        model_weights = model.state_dict()

        n_labels = 0

        if self.kernel_selection == "movement":
            history_weights = self.previous_global_model.state_dict()
            for layer in history_weights.keys():
                if 'conv' in layer:
                    proportion = self.params.conv_rate
                elif 'fc' in layer:
                    proportion = self.params.fc_rate

                candidate_weights[layer] = (model_weights[layer] - history_weights[layer]) * model_weights[layer]
                n_weight = candidate_weights[layer].numel()
                theta = torch.sort(candidate_weights[layer].flatten(), descending=False)[0][int(n_weight * proportion)]
                candidate_weights[layer][candidate_weights[layer] < theta] = 1
                candidate_weights[layer][candidate_weights[layer] != 1] = 0

        return candidate_weights
    

    def inject_handcrafted_filters(self, model, candidate_weights, task, loader):
        conv_weight_names = get_conv_weight_names(model)
        difference = None
        for layer_name, conv_weights in candidate_weights.items():
            if layer_name not in conv_weight_names:
                continue
            model_weights = model.state_dict()
            n_filter = conv_weights.size()[0]
            for i in range(n_filter):
                conv_kernel = model_weights[layer_name][i, ...].clone().detach()
                handcrafted_conv_kernel = self.flip_filter_as_trigger(conv_kernel, difference)
                # handcrafted_conv_kernel = conv_kernel

                mask = conv_weights[i, ...]
                model_weights[layer_name][i, ...] = mask * handcrafted_conv_kernel + (1 - mask) * \
                                                    model_weights[layer_name][i, ...]
                # model_weights[layer_name][i, ...].mul_(1-mask)
                # model_weights[layer_name][i, ...].add_(mask * handcrafted_conv_kernel)

            model.load_state_dict(model_weights)
            difference = self.conv_activation(model, layer_name, task, loader, True) - \
                self.conv_activation(model,layer_name,task,loader,False)

            print("handcraft_conv: {}".format(layer_name))

        torch.cuda.empty_cache()
        if difference is not None:
            feature_difference = self.conv_features(model, task, loader, True) - \
                self.conv_features(model, task, loader,False)
            return feature_difference
    

    def conv_activation(self, model, layer_name, task, loader, attack):
        extractor = FeatureExtractor(model)
        hook = extractor.insert_activation_hook(model)
        module = layer2module(model, layer_name)
        conv_activations = None
        for i, data in enumerate(loader):
            batch = task.get_batch(i, data)
            batch = self.synthesizer.make_backdoor_batch(batch, test=True, attack=attack)
            _ = model(batch.inputs)
            conv_activation = extractor.activations(model, module)
            conv_activation = torch.mean(conv_activation, [0])
            conv_activations = conv_activation if conv_activations is None else conv_activations + conv_activation

        avg_activation = conv_activations / len(loader)
        extractor.release_hooks()
        torch.cuda.empty_cache()
        return avg_activation
    

    def flip_filter_as_trigger(self, conv_kernel: torch.Tensor, difference):
        flip_factor = self.params.flip_factor
        c_min, c_max = conv_kernel.min(), conv_kernel.max()
        pattern = None
        if difference is None:
            pattern_layers, _ = self.synthesizer.get_pattern()
            x_top, y_top = self.synthesizer.x_top, self.synthesizer.y_top
            x_bot, y_bot = self.synthesizer.x_bot, self.synthesizer.y_bot
            pattern = pattern_layers[:, x_top:x_bot, y_top:y_bot]
        else:
            pattern = difference
        w = conv_kernel[0, ...].size()[0]
        resize = transforms.Resize((w, w))
        pattern = resize(pattern)
        p_min, p_max = pattern.min(), pattern.max()
        scaled_pattern = (pattern - p_min) / (p_max - p_min) * (c_max - c_min) + c_min

        crop_mask = torch.sign(scaled_pattern) != torch.sign(conv_kernel)
        conv_kernel = torch.sign(scaled_pattern) * torch.abs(conv_kernel)
        conv_kernel[crop_mask] = conv_kernel[crop_mask] * flip_factor
        return conv_kernel
    

    def conv_features(self, model, task, loader, attack):
        features = None
        if isinstance(model, SimpleNet):
            for i, data in enumerate(loader):
                batch = task.get_batch(i, data)
                batch = self.synthesizer.make_backdoor_batch(batch, test=True, attack=attack)
                feature = model.features(batch.inputs).mean([0])
                features = feature if features is None else features + feature
            avg_features = features / len(loader)
        if isinstance(model, ResNet):
            for i, data in enumerate(loader):
                batch = task.get_batch(i, data)
                batch = self.synthesizer.make_backdoor_batch(batch, test=True, attack=attack)
                feature = model.features(batch.inputs).mean([0])
                features = feature if features is None else features + feature
            avg_features = features / len(loader)

        return avg_features
    

    def fc_activation(self, model, layer_name, task, loader, attack):
        extractor = FeatureExtractor(model)
        hook = extractor.insert_activation_hook(model)
        module = layer2module(model, layer_name)
        neuron_activations = None
        for i, data in enumerate(loader):
            batch = task.get_batch(i, data)
            batch = self.synthesizer.make_backdoor_batch(batch, test=True, attack=attack)
            _ = model(batch.inputs)
            neuron_activation = extractor.activations(model, module)
            neuron_activations = neuron_activation if neuron_activations is None else neuron_activations + neuron_activation

        avg_activation = neuron_activations / len(loader)
        extractor.release_hooks()
        torch.cuda.empty_cache()
        return avg_activation
    

    def optimize_backdoor_trigger(self, model, candidate_weights, task, loader):
        pattern, mask = self.synthesizer.get_pattern()
        pattern.requires_grad = True

        x_top, y_top = self.synthesizer.x_top, self.synthesizer.y_top
        x_bot, y_bot = self.synthesizer.x_bot, self.synthesizer.y_bot

        cbots, ctops = list(), list()
        for h in range(pattern.size()[0]):
            cbot = (0 - task.means[h]) / task.lvars[h]
            ctop = (1 - task.means[h]) / task.lvars[h]
            cbots.append(round(cbot, 2))
            ctops.append(round(ctop, 2))

        raw_weights = deepcopy(model.state_dict())
        self.set_handcrafted_filters2(model, candidate_weights, "conv1.weight")
        for epoch in range(2):
            losses = list()
            for i, data in enumerate(loader):
                batch_size = self.params.batch_size

                clean_batch, backdoor_batch = task.get_batch(i, data), task.get_batch(i, data)

                backdoor_batch.inputs[:batch_size] = (1 - mask) * backdoor_batch.inputs[:batch_size] + mask * pattern
                backdoor_batch.labels[:batch_size].fill_(self.params.backdoor_label)

                self.set_handcrafted_filters2(model, candidate_weights, "conv1.weight")

                # loss, grads = trigger_attention_loss(raw_model, model, backdoor_batch.inputs, pattern, grads=True)
                loss, grads = trigger_loss(model, backdoor_batch.inputs, clean_batch.inputs, pattern, grads=True)
                losses.append(loss.item())

                pattern = pattern + grads[0] * 0.1

                n_channel = pattern.size()[0]
                for h in range(n_channel):
                    pattern[h, x_top:x_bot, y_top:y_bot] = torch.clamp(pattern[h, x_top:x_bot, y_top:y_bot], cbots[h],
                                                                       ctops[h], out=None)

                model.zero_grad()
            print("epoch:{} trigger loss:{}".format(epoch, np.mean(losses)))

        print(pattern[0, x_top:x_bot, y_top:y_bot].cpu().data)

        self.synthesizer.pattern = pattern.clone().detach()
        self.synthesizer.pattern_tensor = pattern[x_top:x_bot, y_top:y_bot]

        model.load_state_dict(raw_weights)
        torch.cuda.empty_cache()
    

    def inject_handcrafted_neurons(self, model, candidate_weights, task, diff, loader):
        handcrafted_connectvites = defaultdict(list)
        target_label = self.params.backdoor_label
        n_labels = -1
        if isinstance(task, Cifar10FederatedTask):
            n_labels = 10
        elif isinstance(task, TinyImagenetFederatedTask):
            n_labels = 200

        fc_names = get_neuron_weight_names(model)
        fc_diff = diff
        last_layer, last_ids = None, list()
        for layer_name, connectives in candidate_weights.items():
            if layer_name not in fc_names:
                continue
            raw_model = deepcopy(model)
            model_weights = model.state_dict()
            ideal_signs = torch.sign(fc_diff)
            n_next_neurons = connectives.size()[0]
            # last_layer
            if n_next_neurons == n_labels:
                break

            ideal_signs = ideal_signs.repeat(n_next_neurons, 1) * connectives
            # count the flip num
            n_flip = torch.sum(((ideal_signs * torch.sign(model_weights[layer_name]) * connectives == -1).int()))
            print("n_flip in {}:{}".format(layer_name, n_flip))
            model_weights[layer_name] = (1 - connectives) * model_weights[layer_name] + torch.abs(
                connectives * model_weights[layer_name]) * ideal_signs
            model.load_state_dict(model_weights)
            last_layer = layer_name
            fc_diff = self.fc_activation(model, layer_name, task, loader, attack=True).mean([0]) - self.fc_activation(
                model, layer_name, task, loader, attack=False).mean([0])