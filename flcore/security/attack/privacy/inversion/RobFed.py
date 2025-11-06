
import copy
import torch

from flcore.models.model_utils import BaseModel
from flcore.security.attack.privacy.inversion.base import BaseInversionAttack
from flcore.security.attack.privacy.inversion.utils.imprint_block import ImprintBlock, SparseImprintBlock, OneShotBlock, OneShotBlockSparse




block_dict = {"ImprintBlock": ImprintBlock,
              "SparseImprintBlock": SparseImprintBlock,
              "OneShotBlock": OneShotBlock,
              "OneShotBlockSparse": OneShotBlockSparse
              }

class ReviseModel(BaseModel):  
    def __init__(self, block, model):
        super().__init__()
        self.block = block
        self.model = model

    def forward(self, x):
        x = self.block(x) 
        x = self.model(x)  
        return x

class RobFedAttack(BaseInversionAttack):
# Labels are not required for this attack
    def __init__(self, Server, config):
        super().__init__('analytic', 'RobFed', Server, \
                         config, conf_path='./security/attack/privacy/config/RobFed.yaml')
        self.block_fn = block_dict[self.config['model_modification_type']]
        self.position = self.config['model_modification']['position']
        self.mean = torch.as_tensor(self.mean)[None, :, None, None]
        self.std = torch.as_tensor(self.std)[None, :, None, None]


    def reconstruction(self, shared_data, labels):
        """This is somewhat hard-coded for images, but that is not a necessity."""
        # if metadata.modality == "text": not completed
        # Initialize stats module for later usage:
        if "ImprintBlock" in self.server_secrets.keys():
            weight_idx = self.server_secrets["ImprintBlock"]["weight_idx"]
            bias_idx = self.server_secrets["ImprintBlock"]["bias_idx"]
        else:
            raise ValueError(f"No imprint hidden in model according to server.")

        target_updates=[]
        for i in range(len(shared_data["updates"])):
            uptensor = shared_data["parameters"][i] - shared_data["updates"][i]
            target_updates.append(uptensor)
        bias_grad = target_updates[bias_idx].clone()
        weight_grad = target_updates[weight_idx].clone()

        if self.config['sort_by_bias']:
            # This variant can recover from shuffled rows under the assumption that biases would be ordered
            # server_payload: server model params
            _, order = shared_data["updates"][1].sort(descending=True)
            bias_grad = bias_grad[order]
            weight_grad = weight_grad[order]

        if self.server_secrets["ImprintBlock"]["structure"] == "cumulative":
            for i in reversed(list(range(1, weight_grad.shape[0]))):
                weight_grad[i] -= weight_grad[i - 1]
                bias_grad[i] -= bias_grad[i - 1]

        # This is the attack:
        layer_inputs = self.invert_fc_layer(weight_grad, bias_grad, [])

        # Reduce hits if necessary:
        layer_inputs = self.reduce_hits(layer_inputs, weight_grad, bias_grad, shared_data)

        # Reshape images, re-identify token embeddings:
        reconstructed_inputs = self.reformat_data(layer_inputs, shared_data)

        return reconstructed_inputs, labels

    def place_malicious_block(self, modified_model, num_clients):
        """The block is placed directly before the named module given by "position".
        If none is given, the block is placed before the first layer.
        """
        CANDIDATE_FIRST_LAYERS = (
        torch.nn.Linear,
        torch.nn.Flatten,
        torch.nn.Conv2d)
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
                print(f"Block inserted at feature shape {data_shape}.")
                module_to_be_modified = module
                block_found = True
                break

        if not block_found:
            raise ValueError(f"Could not find module {self.position} in model to insert layer.")

        # Insert malicious block:
        block = self.block_fn(data_shape, self.config['num_bins'], connection=self.config['connection'], linfunc=self.config['model_modification']['linfunc'], mode=self.config['mode'])
        replacement = ReviseModel(block, module_to_be_modified)
        self.replace_module_by_instance(modified_model, module_to_be_modified, replacement)
        for idx, param in enumerate(modified_model.parameters()):
            if param is block.linear0.weight:
                weight_idx = idx
            if param is block.linear0.bias:
                bias_idx = idx
        self.server_secrets = dict(ImprintBlock=dict(weight_idx=weight_idx, bias_idx=bias_idx, shape=data_shape, structure=block.structure))

        if self.position is not None:
            if self.config['model_modification_type'] == "SparseImprintBlock":
                block_fn = type(None)  # Linearize the full model for SparseImprint
            if self.config['model_modification']['handle_preceding_layers'] == "identity":
                self._linearize_up_to_imprint(modified_model, block_fn)
            elif self.config['model_modification']['handle_preceding_layers'] == "VAE":
                # Train preceding layers to be a VAE up to the target dimension
                # zm:not complete self.train_encoder_decoder
                modified_model, decoder = self.train_encoder_decoder(modified_model, block_fn)
                self.server_secrets["ImprintBlock"]["decoder"] = decoder
            else:
                # Otherwise do not modify the preceding layers. The attack then returns the layer input at this position directly
                pass

        # Reduce failures in later layers:
        # Note that this clashes with the VAE option!
        # zm del:self._normalize_throughput
        self.model = [copy.deepcopy(modified_model) for i in range(num_clients)]
        return self.model 

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

        throughput = torch.zeros([1, *self.input_data_shape], **setup)
        model(throughput)
        [h.remove() for h in hooks_list]
        return feature_shapes

    def _linearize_up_to_imprint(self, model, block_fn):
        """This linearization option only works for a ResNet architecture."""
        first_conv_set = False  # todo: make this nice
        for name, module in model.named_modules():
            if isinstance(module, block_fn):
                break
            with torch.no_grad():
                if isinstance(module, torch.nn.BatchNorm2d):
                    # module.weight.data = (module.running_var.data.clone() + module.eps).sqrt()
                    # module.bias.data = module.running_mean.data.clone()
                    torch.nn.init.ones_(module.running_var)
                    torch.nn.init.ones_(module.weight)
                    torch.nn.init.zeros_(module.running_mean)
                    torch.nn.init.zeros_(module.bias)
                if isinstance(module, torch.nn.Conv2d):
                    if not first_conv_set:
                        torch.nn.init.dirac_(module.weight)
                        num_groups = module.out_channels // 3
                        module.weight.data[: num_groups * 3] = torch.cat(
                            [module.weight.data[:3, :3, :, :]] * num_groups
                        )
                        first_conv_set = True
                    else:
                        torch.nn.init.zeros_(module.weight)  # this is the resnet rule
                if "downsample.0" in name:
                    torch.nn.init.dirac_(module.weight)
                    num_groups = module.out_channels // module.in_channels
                    concat = torch.cat(
                        [module.weight.data[: module.in_channels, : module.in_channels, :, :]] * num_groups
                    )
                    module.weight.data[: num_groups * module.in_channels] = concat
                if isinstance(module, torch.nn.ReLU):
                    self.replace_module_by_instance(model, module, torch.nn.Identity())

    def invert_fc_layer(self, weight_grad, bias_grad, image_positions):
        """The basic trick to invert a FC layer."""
        # By the way the labels are exactly at (bias_grad < 0).nonzero() if they are unique
        valid_classes = bias_grad != 0
        intermediates = weight_grad[valid_classes, :] / bias_grad[valid_classes, None]
        if len(image_positions) == 0:
            reconstruction_data = intermediates
        elif len(image_positions) == 1:
            reconstruction_data = intermediates.mean(dim=0)
        else:
            reconstruction_data = intermediates[image_positions]
        return reconstruction_data

    def reduce_hits(self, layer_inputs, weight_grad, bias_grad, shared_data):
        """In case of numerical instability or gradient noise, more bins can be hit than expected."""
        len_data = self.num_images  # Not strictly needed for the attack, used to pad/trim
        if len_data >= layer_inputs.shape[0]:
            print("You can recovery picture no more than",layer_inputs.shape[0])
            self.Server.logger.log(round=0, identity="Server", action="Preparing for recovery", message=f"You can recovery picture no more than{layer_inputs.shape[0]}")
            # Fill up with zero if not enough data can be found?
            if self.config['breach_padding']:
                missing_entries = layer_inputs.new_zeros(len_data - layer_inputs.shape[0], *layer_inputs.shape[1:])
                layer_inputs = torch.cat([layer_inputs, missing_entries], dim=0)
        else:
            # Cut additional hits:
            if self.config['breach_reduction'] == "bias":
                # this rule is optimal for clean data with few bins:
                best_guesses = torch.topk(bias_grad[bias_grad != 0].abs(), len_data, largest=True)[1]
            elif self.config['breach_reduction'] == "weight":
                # this rule is best when faced with differential privacy:
                best_guesses = torch.topk(weight_grad.mean(dim=1)[bias_grad != 0].abs(), len_data, largest=True)[1]
            else:  # None #
                # Warning: This option can mess up metrics later on (as more data is recpnstructed than exists)
                best_guesses = torch.arange(layer_inputs.shape[0])

            layer_inputs = layer_inputs[best_guesses]
        return layer_inputs

    def reformat_data(self, layer_inputs, shared_data):
        """After the actual attack has happened, we have to do some work to piece everything back together in the
        desired data format."""

        data_shape = self.server_secrets["ImprintBlock"]["shape"]

        if "decoder" in self.server_secrets["ImprintBlock"].keys():
            inputs = self.server_secrets["ImprintBlock"]["decoder"](layer_inputs)

        if self.config['modality'] == "vision":
            inputs = layer_inputs.reshape(layer_inputs.shape[0], *data_shape)[:, :3, :, :]
            if inputs.shape[1:] != self.input_data_shape:
                interp_mode = dict(mode="bicubic", align_corners=False)
                inputs = torch.nn.functional.interpolate(inputs, size=self.input_data_shape[1:], **interp_mode)
            #inputs = torch.max(torch.min(inputs, (1 - self.mean) / self.std), -self.mean / self.std)
        else:
            pass
            # not completed
            '''
            inputs = layer_inputs.reshape(layer_inputs.shape[0], *data_shape)
            if self.cfg.token_strategy is not None:
                leaked_tokens = self.recover_token_information(shared_data, server_payload, rec_models[0].name)
            inputs = self._postprocess_text_data(dict(data=inputs, labels=leaked_tokens), models=rec_models)["data"]
            '''
        return inputs