import torch
import torch.nn as nn

def TV(data, scale):
    scale = float(scale)
    dx = torch.mean(torch.abs(data[:, :, :, :-1] - data[:, :, :, 1:]))
    dy = torch.mean(torch.abs(data[:, :, :-1, :] - data[:, :, 1:, :]))
    """Use TV from breach
    grad_weight = torch.tensor([[0, 0, 0], [0, -1, 1], [0, 0, 0]], device = device_, dtype = dtype_).unsqueeze(0).unsqueeze(1)
    grad_weight = torch.cat((torch.transpose(grad_weight, 2, 3), grad_weight), 0)
    grad_weight = torch.cat([grad_weight] * 3, 0)
    diffs = torch.nn.functional.conv2d(
        data, grad_weight, None, stride=1, padding=1, dilation=1, groups=3
    )
    squares = diffs.abs() + 1e-8
    squared_sums = squares[:, 0::2] + squares[:, 1::2]
    TV_loss = squared_sums.mean().reshape((1,)) * scale"""
    return (dx + dy) * scale

def L2(data, scale):
    pnorm = 2
    scale = float(scale)
    return scale * torch.pow(torch.mean(torch.pow(data, pnorm)), 1./pnorm)


def _get_hook_for_bn(name, bn_layers_dict):
    def hook(model, input, output):
        bn_layers_dict[name] = input[0]
    return hook

def collect_bn_layers(model):
    batch_norm_layers = {}
    norm_inputs = {}

    # 遍历模型的所有子模块
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            batch_norm_layers[name] = module
            module.register_forward_hook(_get_hook_for_bn(name, norm_inputs))
    
    return batch_norm_layers, norm_inputs

def BN(bn_layers, norm_inputs):
    """Computes the batch normalization regularizer loss.

    Args:
        bn_layers (dict): dict of batch normalization layers.

    Returns:
        torch.Tensor: The batch normalization regularizer loss.
    """
    bn_reg = 0
    for name, layer in bn_layers.items():
        fm = norm_inputs[name]
        if len(fm.shape) == 3:
            dim = [0, 2]
        elif len(fm.shape) == 4:
            dim = [0, 2, 3]
        elif len(fm.shape) == 5:
            dim = [0, 2, 3, 4]
        bn_reg += torch.norm(fm.mean(dim=dim) - layer.running_mean, p=2)
        bn_reg += torch.norm(fm.var(dim=dim) - layer.running_var, p=2)
    return bn_reg

def group_attack(received_gradients, batch_size=64):
    """Multiple simultaneous attacks with different random states

    Args:
        received_gradients: the list of gradients received from the client.
        batch_size: batch size.

    Returns:
        a tuple of the best reconstructed images and corresponding labels
    """
    group_fake_x = []
    group_fake_label = []
    group_optimizer = []

    for _ in range(self.group_num):
        fake_x, fake_label, optimizer = _setup_attack(
            self.x_shape,
            self.y_shape,
            self.optimizer_class,
            self.optimize_label,
            self.pos_of_final_fc_layer,
            self.device,
            received_gradients,
            batch_size,
            **self.kwargs,
        )

        group_fake_x.append(fake_x)
        group_fake_label.append(fake_label)
        group_optimizer.append(optimizer)

    best_distance = [float("inf") for _ in range(self.group_num)]
    best_fake_x = [x_.detach().clone() for x_ in group_fake_x]
    best_iteration = [0 for _ in range(self.group_num)]

    for i in range(1, self.num_iteration + 1):
        for worker_id in range(self.group_num):
            self.reset_seed(self.group_seed[worker_id])
            closure = self._setup_closure(
                group_optimizer[worker_id],
                group_fake_x[worker_id],
                group_fake_label[worker_id],
                received_gradients,
            )
            distance = group_optimizer[worker_id].step(closure)

            if best_distance[worker_id] > distance:
                best_fake_x[worker_id] = group_fake_x[worker_id].detach().clone()
                best_distance[worker_id] = distance
                best_iteration[worker_id] = i

    return best_fake_x

def GC(x, group_x):
    """Computes the group consistency loss between an input and a group of inputs.

    Args:
        x (torch.Tensor): The input tensor.
        group_x (list): List of tensors representing the group.

    Returns:
        torch.Tensor: The group consistency loss.
    """
    mean_group_x = sum(group_x) / len(group_x)
    return torch.norm(x - mean_group_x, p=2)

RegFn_list = ['TV', 'L2', 'BN', 'GC']