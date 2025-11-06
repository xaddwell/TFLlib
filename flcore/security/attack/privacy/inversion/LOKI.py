

# from statistics import NormalDist
import copy
import torch
import numpy as np
import torch.nn as nn
import torch.functional as F

from flcore.models.model_utils import BaseModel
from flcore.security.attack.privacy.inversion.base import BaseInversionAttack
from flcore.security.attack.privacy.inversion.utils.imprint_block import SparseImprintLayer



class ReviseModel(BaseModel):  
    def __init__(self, block, model):
        super().__init__()
        self.block = block
        self.model = model

    def forward(self, x):
        x = self.block(x) 
        x = self.model(x)  
        return x


class LOKIAttack(BaseInversionAttack):
    def __init__(self, Server, config):
        super().__init__('analytic', 'LOKI', Server, \
                         config, conf_path='./security/attack/privacy/config/LOKI.yaml')
        
        self.block_fn = SparseImprintLayer
        self.position = self.config['model_modification']['position']
        self.linfunc = self.config['model_modification']['linfunc']
        self.data_ch, *self.im_dim = self.input_data_shape # 3 [32, 32] for CIFAR10
        self.myvalue = self.data_ch * self.im_dim[0] * self.im_dim[1]
        self.num_bins = self.config['batch_size']*4

    def place_malicious_block(self, modified_model, num_clients):
        CANDIDATE_FIRST_LAYERS = (
            torch.nn.Linear,
            torch.nn.Flatten,
            torch.nn.Conv2d
        )

        self.num_clients = num_clients
        self.conv_sizes = self.num_clients*self.data_ch

        modified_models = []
        for i in range(num_clients):
            if self.position is None:
                for name, module in modified_model.named_modules():
                    if isinstance(module, CANDIDATE_FIRST_LAYERS):
                        print(f"First layer determined to be {name}")
                        self.position = name
                        break

            block_found = False
            for name, module in modified_model.named_modules():
                if self.position in name:  # give some leeway for additional containers.
                    feature_shapes = self._introspect_model(modified_model)
                    data_shape = feature_shapes[name]["shape"][1:]
                    print(f"Block inserted at feature shape {data_shape}")
                    module_to_be_modified = module
                    block_found = True
                    break

            if not block_found:
                raise ValueError(f"Could not find module {self.position} in model to insert layer.")

            block = self.block_fn(self.num_bins, i, self.data_ch, self.im_dim, self.config['conv_scale'], self.conv_sizes, self.mean, self.std, self.linfunc)

            
            
            # replacement = torch.nn.Sequential(block, module_to_be_modified)
            # tmp_model = self.replace_module_by_instance(copy.deepcopy(modified_model), module_to_be_modified, replacement)

            tmp_model = ReviseModel(copy.deepcopy(block),copy.deepcopy(modified_model))

            if i == self.cid:
                for idx, param in enumerate(tmp_model.named_parameters()):
                    name, tensor = param
                    if 'FC1.weight' in name:
                        weight_idx = idx
                    elif 'FC1.bias' in name:
                        bias_idx = idx
                print(f"weight_idx is {weight_idx}, bias_idx is {bias_idx}")
                self.server_secrets=dict(ImprintBlock=dict(weight_idx=weight_idx, bias_idx=bias_idx, shape=data_shape))
            modified_models.append(tmp_model)
        
        return modified_models

    def reconstruction(self, shared_data, labels):
        count = 0
        recon_corresp = []
        #shared_data['parameters']
        grad_w, grad_b = self._get_params(shared_data['parameters'], shared_data['updates'])

        for m in range(grad_w.size()[0]):
            if grad_w[m].abs().sum() != 0:
                count += 1
            if grad_w[m].abs().sum() != 0 and (grad_w[m]-grad_w[m][0]).abs().sum() != 0:
                recon_corresp.append(self._norm_image(grad_w[m]).reshape(self.data_ch,self.im_dim[0],self.im_dim[1]))
        print("Total hits is:", len(recon_corresp))
        stacked_recon = torch.stack(recon_corresp, dim=0)
        return stacked_recon, labels

    def _get_params(self, original_net, my_model):
        weight_idx = self.server_secrets["ImprintBlock"]["weight_idx"]
        bias_idx = self.server_secrets["ImprintBlock"]["bias_idx"]

        # 将a和b中的内部列表转换为元组
        a_tuples = [tuple(inner_list) for inner_list in original_net[weight_idx].tolist()]
        b_tuples = [tuple(inner_list) for inner_list in my_model[weight_idx].tolist()]
        # 使用元组执行对称差集操作
        x = list(set(b_tuples) - set(a_tuples))

        # 如果您需要将结果x中的元素再次转换为列表（为了进一步处理或与代码中的其他部分保持一致）
        x_lists = [list(inner_tuple) for inner_tuple in x]
        #print(x_lists)

        if self.config['num_client_scaling']:
            grad_w = (((original_net[weight_idx] * (self.num_clients-1)) + my_model[weight_idx])/self.num_clients - original_net[weight_idx])[:, self.myvalue*self.cid: self.myvalue*(self.cid+1)]
            grad_b = (((original_net[bias_idx] * (self.num_clients-1)) + my_model[bias_idx])/self.num_clients - original_net[bias_idx])
        else:
            pass

        return grad_w, grad_b

    def _norm_image(self, image):
        x = copy.deepcopy(image)
        x = torch.abs(x)
        x = x - min(x)
        x = x/max(x)
        return x
       
    def _introspect_model(self, model, modality="vision"):
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
                feature_shapes[name] = dict(shape=input[0].shape, info=str(module))

            return hook_fn

        hooks_list = []
        for name, module in model.named_modules():
            #export the input of register moduler
            hooks_list.append(module.register_forward_hook(named_hook(name)))

        throughput = torch.zeros([1,  self.data_ch, *self.im_dim], **setup)
        print(throughput.shape)
        model(throughput)
        [h.remove() for h in hooks_list]
        return feature_shapes