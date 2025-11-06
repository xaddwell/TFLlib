import os
import yaml
import random
import torch
from torchvision.utils import make_grid
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC
from typing import Tuple 
from flcore.security.attack.privacy.inversion.utils.metrics_torch import cal_lpips, psnr_compute, cw_ssim



MEAN_STD = {
    'CIFAR10': [[0.4914672374725342, 0.4822617471218109, 0.4467701315879822], \
        [0.24833508, 0.24547848, 0.26617324]],
    'CIFAR100': [[0.5071, 0.4865, 0.4409], [0.1942, 0.1918, 0.1958]],
    'FEMNIST': [[0.9643], [0.1487]]
}

class BaseInversionAttack(ABC):
    
    def __init__(self, attack_type, attack_name, Server, config, conf_path=None):
        self.attack_type = attack_type
        self.attack_name = attack_name
        self.mean, self.std = MEAN_STD[config.data_name]
        self.num_clients = Server.current_num_join_clients
        self.Server = Server
        self.num_images = self.Server.batch_size
        self.input_data_shape = Server.train_data[0][0][0][0].size()
        if conf_path:
            with open(conf_path, 'r', encoding='utf-8') as f:
                self.config = yaml.load(f.read(), Loader=yaml.FullLoader)
                self.cid = self.config['cid']

            if config:
                # merge a configparse and a yaml_dict
                self.config = {**self.config, **vars(config)}
        else:
            self.config = config
        if 'attack_mode' not in self.config:
            self.config['attack_mode'] = 'Passive'
        if self.cid is None:
            self.cid = random.randint(0, Server.conf.num_clients-1)
        if type(self.cid) == Tuple:
            self.cid = self.cid[0]
        self.config["attack_type"] = 'inversion'
        self.Server.logger.log(round=-1, identity="Server", action="Attack", message=f"Initializing reconstruction attacker")

    def preprocess(self, updates):
        pass

    def reconstruction(self, shared_info, labels):
        raise NotImplementedError("No reconstruction attack method completed")

    def label_inference(self, shared_info):
        raise NotImplementedError("No label inference attack method completed")

    def replace_module_by_instance(self, model, old_module, replacement):
        def replace(model):
            for child_name, child in model.named_children():
                if child is old_module:
                    setattr(model, child_name, replacement)
                else:
                    replace(child)
        replace(model)
        return model

    def visual_image(self, x, label=None, filename ="res_img.png"):
        # zm: incomplete label->class
        current_work_dir = os.path.dirname(__file__)
        save_path = os.path.join(current_work_dir, 'result_img', filename)
        if x.shape[0] == 1:
            plt.axis("off")
            plt.imshow(x[0].permute(1, 2, 0).cpu())
            if label is not None:
                plt.title(f"Data with label {label}")
        else:
            grid_shape = int(torch.as_tensor(x.shape[0]).sqrt().ceil())
            s = 24 if x.shape[3] > 150 else 6
            fig, axes = plt.subplots(grid_shape, grid_shape, figsize=(s, s))
            label_classes = []
            for i, (im, axis) in enumerate(zip(x, axes.flatten())):
                axis.imshow(im.permute(1, 2, 0).cpu())
                if label is not None:
                    label_classes.append(label[i])
                axis.axis("off")
            if label is not None:
                print(label_classes) 
                
        plt.savefig(save_path) 
        plt.close()

    from typing import List
    def visual_imglist(self, x: List, labels: List, dirname="gt_img", filename="tr_img.png"):
        if os.path.exists(dirname) == False:
            os.mkdir(dirname)
        # 取一个batch的数据进行恢复
        x = x[:self.num_images]
        labels = labels[:self.num_images]
        # 转换为tensor格式
        x = torch.stack(x)
        images_tensor = torch.squeeze(x).cpu()
        
        # 将归一化的图片数据恢复到0-255
        images_tensor = images_tensor * 255
        images_tensor = images_tensor.type(torch.uint8) 

        # 计算需要多少行来显示图片
        num_images = len(x)
        num_rows = min(4, num_images)

        # 每16张图片创建一个图像网格
        for i in range(0, num_images, 16):
            images_batch = images_tensor[i:min(i+16, num_images)]
            labels_batch = labels[i:min(i+16, num_images)]
            # 单通道图片
            if len(images_batch.shape) == 3:
                images_batch = images_batch.unsqueeze(dim=1)
            # 根据当前批次图片的数量重新计算行列数
            current_batch_size = len(images_batch)
            rows = min(4, current_batch_size)
            cols = min(4, max(1, current_batch_size // rows))

            # 创建网格
            grid_img = make_grid(images_batch, nrow=cols, padding=2)

            # 转换为numpy格式绘图
            npimg = grid_img.numpy()
            fig, ax = plt.subplots(figsize=(cols * 3, rows * 3))
            # 隐藏坐标轴
            ax.axis('off')
            # 绘制图片，由于图片已经是0-255，直接绘制
            ax.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')

            # 添加标签
            padding = 2
            img_size = 32
            for img_idx, label in enumerate(labels_batch):
                if torch.is_tensor(label):
                    label = label.cpu().numpy()
              # 计算标签的位置
                x_pos = ((img_idx % cols) + 1) * (img_size + padding) - padding
                y_pos = ((img_idx // cols) + 1) * (img_size + padding) - padding
                ax.text(x_pos, y_pos, str(label),
                        color='white', fontsize=12, ha='right', va='bottom', backgroundcolor='black')
            # 保存或显示图片
            #current_work_dir = os.path.dirname(__file__)
            save_path = os.path.join(dirname, filename)
            plt.savefig(f"{save_path.rsplit('.', 1)[0]}_{i//16}.png")
            plt.show()
            plt.close()

    def metrics(self, recover_batch, ori_batch, net='vgg'):
        psnr = psnr_compute(recover_batch, ori_batch)
        lpip = cal_lpips(recover_batch, ori_batch, net=net)
        ssim = cw_ssim(recover_batch, ori_batch)
        metrics_dict = {'psnr': psnr, 'lpip': lpip, 'ssim':ssim}
        print(metrics_dict)
        self.Server.logger.log(round=-1, identity="Server", action="Attack", message=f"metrics: {metrics_dict}")
        return metrics_dict