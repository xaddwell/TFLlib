
import torch
import torch.nn as nn
import torch.nn.functional as F

import copy
import numpy as np


from flcore.security.attack.privacy.inference.mia.base import MembershipInferenceAttack


DATA_TYPE = {
    "resnet18": "vision",
    "logreg": "text",
    "tinybert": "text",
}

class Classifier(nn.Module):
    def __init__(self, n_inputs, numneurons = 2000):
        super(Classifier, self).__init__()
        self.data_shape = n_inputs
        self.data_size = torch.prod(torch.as_tensor(n_inputs))
        self.fc_1 = nn.Linear(self.data_size, numneurons)
        self.fc_attack = nn.Linear(numneurons, self.data_size)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc_1(x)
        x = F.relu(x)
        x = self.fc_attack(x)
        return x.unflatten(dim=1, sizes=self.data_shape)

class AMIAttacker(MembershipInferenceAttack):
    def __init__(self, Server, conf2=None):
        conf_path = None
        super().__init__("AMI", Server, conf_path, conf2)
        self.num_clients = self.config['num_clients']
        self.config['attack_mode'] = 'Active'
        self.config['metrics'] = {'acc': 'per_class', 'precision_recall': 'per_class'}

    def preprocess(self, mem_train, nonmem_train):
        batch_size=self.Server.batch_size
        data_len = min(len(mem_train), len(nonmem_train))
        mem_train = list(mem_train)[:data_len]
        nonmem_train = list(nonmem_train)[:data_len]
        for i in range(len(mem_train)):
            mem_train[i] = mem_train[i] + (1,)
        for i in range(len(nonmem_train)):
            nonmem_train[i] = nonmem_train[i] + (0,)
        attack_data = mem_train + nonmem_train
        self.attack_data_loader = torch.utils.data.DataLoader(
            attack_data, batch_size=batch_size, shuffle=True, num_workers=2)

    def MIA(self, target):
        self.preprocess(target, self.Server.test_data) 
        self.target_model = self.Server.clients[self.cid].model.to(self.device)
        layer = self._find_module(self.target_model, "fc_attack")
        outlist, c, y = [], [], []

        # 定义一个容器来保存中间输出
        intermediate_outputs = []
        # 定义一个钩子函数
        def get_intermediate_output(module, input, output):
            intermediate_outputs.append(output)
        # 注册钩子
        hook = layer.register_forward_hook(get_intermediate_output)

        for batch in self.attack_data_loader:
            if(len(batch) == 3):
                inputs, targets, members = batch
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                output = self.target_model(inputs.to(self.device))
            else:
                inputs, masks, targets, members = batch
                inputs, masks, targets = inputs.to(self.device), masks.to(self.device), targets.to(self.device)
                output = self.target_model(inputs.to(self.device), masks).logits

            y.extend(targets.squeeze().detach().clone().cpu().tolist())
            c.extend(members.detach().clone().cpu().tolist())

        for i in range(len(intermediate_outputs)):
            '''
            p_t = []
            for j in range(intermediate_outputs[0].shape[0]):
                t = intermediate_outputs[i][:, j] > 0
                p_t.append(t)
            stacked_tensors = torch.stack(p_t, dim=0)
            result_tensor = torch.any(stacked_tensors, dim=0)
            '''
            result_tensor = intermediate_outputs[i][:, 0] > 0
            out = result_tensor.detach().clone().cpu().tolist()
            out = [int(item) for item in out]
            outlist.extend(out)
        # 不要忘了在完成后移除钩子，以避免内存泄露
        hook.remove()
        return np.array(outlist), np.array(c), np.array(y)

    def place_malicious_block(self, ori_model, num_clients):
        CANDIDATE_FIRST_LAYERS = (
        torch.nn.Linear,
        torch.nn.Flatten,
        torch.nn.Conv2d)

        modified_model = copy.deepcopy(ori_model)

        for name, module in modified_model.named_modules():
            if isinstance(module, CANDIDATE_FIRST_LAYERS):
                print(f"First layer determined to be {name}")
                self.position = name
                break

        for name, module in modified_model.named_modules():
            if self.position in name:  # give some leeway for additional containers.
                feature_shapes = self._introspect_model(modified_model)
                data_shape = feature_shapes[name]["shape"][1:]
                print(f"Block inserted at feature shape {data_shape}")
                module_to_be_modified = module
                break

        block = Classifier(data_shape)
        replacement = torch.nn.Sequential(block, module_to_be_modified)
        replacement.train()
        self.replace_module_by_instance(modified_model, module_to_be_modified, replacement)
        modified_models = []

        for i in range(self.num_clients):
            modified_models.append(copy.deepcopy(modified_model))
        return modified_models
    
    def _introspect_model(self, model):
        modality = DATA_TYPE.get(model.__class__.__name__.lower(), 'vision')
        """Compute model feature shapes."""
        feature_shapes = dict()
        if modality == "vision":
            setup = dict(device=next(iter(model.parameters())).device, dtype=next(iter(model.parameters())).dtype)
        elif modality == "text":
            setup = dict(device=next(iter(model.parameters())).device, dtype=torch.long)
        else:
            raise ValueError(f"Invalid modality {modality} for model introspection.")

        def named_hook(name):
            def hook_fn(module, input, output):
                if input is not tuple():
                    feature_shapes[name] = dict(shape=input[0].shape, info=str(module))

            return hook_fn

        hooks_list = []
        for name, module in model.named_modules():
            #export the input of register moduler
            hooks_list.append(module.register_forward_hook(named_hook(name)))
        if modality == "vision":
            self.data_ch, *self.im_dim = self.Server.train_data[0][0][0][0].size()
            throughput = torch.zeros([1,  self.data_ch, *self.im_dim], **setup)
            model(throughput)
        else:
            throughput = self.Server.test_data[0][0].unsqueeze(0).to(setup["device"])
            mask = self.Server.test_data[0][1].unsqueeze(0).to(setup["device"])
            model(throughput, mask)
        [h.remove() for h in hooks_list]
        return feature_shapes
    
    def replace_module_by_instance(self, model, old_module, replacement):
        def replace(model):
            for child_name, child in model.named_children():
                if child is old_module:
                    setattr(model, child_name, replacement)
                else:
                    replace(child)
        replace(model)

    def _find_module(self, model, name):
        # 如果在当前模型的直接子模块中找到了该层，则返回该层
        if name in model._modules:
            return model._modules[name]
        # 否则，遍历所有子模块
        for child_name, child_module in model.named_children():
            layer = self._find_module(child_module, name)
            if layer:
                return layer
        return None