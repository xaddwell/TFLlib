import copy
import time
import torch.nn.functional as F

from flcore.security.attack.privacy.inversion.base import BaseInversionAttack
from flcore.security.attack.privacy.inversion.utils.objective_fn import *
from flcore.security.attack.privacy.inversion.utils.regularization import *

# data: self._client_uploads MODEL:models DATA_SIZE:weights CLIENT_METRICS
class OptimizationBasedAttack(BaseInversionAttack):
    # zm: image mean std not completed
    def __init__(self, Server, config):
        conf_path = './security/attack/privacy/config/'+config.privacy_attack+'.yaml'
        super().__init__("Inversion", config.privacy_attack, Server, config, conf_path)
        if self.config['cost_fn'] not in ObjectFn_list:
            raise ValueError(f"Unknown cost function {config.cost_fn} is given")
        self.cost_fn = self.config['cost_fn']
        self.reg = {}

        if 'regularization' in self.config and self.config['regularization'] is not None:
            for k in self.config['regularization'].keys():
                if k in RegFn_list:
                    self.reg[k] = self.config['regularization'][k]
                else:
                    raise ValueError(f"Unknown reg method {k} is given")

        self.scheduler = self.config['scheduler']

        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
        self.reconstruct_label = False
        print('*'*10, 'Attack config for OptBasedReconstruction Attack', '*'*10)

        self.Server.logger.log(round=-1, identity="Server", action="Attack", message=f"metrics: Attack config for OptBasedReconstruction Attack\n{self.config}")
        print(self.config)
        self.num_classes = self.config['num_classes']
        self.dtype = torch.float
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    def _init_images(self, img_shape):
        if self.config['init_img'] == 'randn':
            img = torch.randn((self.num_images, *img_shape), device = self.device, dtype = self.dtype)
        elif self.config['init_img'] == 'rand':
            img = torch.rand((self.num_images, *img_shape), device = self.device, dtype = self.dtype) 
        elif self.config['init_img'] == 'randint':
            img = torch.randint(255, (self.num_images, *img_shape), device = self.device, dtype = self.dtype)
        elif self.config['init_img'] == 'zeros':
            img = torch.zeros((self.num_images, *img_shape), device = self.device, dtype = self.dtype)
        else:
            raise ValueError()   
        img.requires_grad = True
        img.grad = torch.zeros_like(img)
        return img

    def _gradient_closure(self, optimizer, x_trial, input_gradient, label):
        def closure():
            #print("input grad: ",  input_gradient.shape)
            optimizer.zero_grad()
            self.model.zero_grad()
            loss = self.loss_fn(self.model(x_trial), label)
            gradient = torch.autograd.grad(loss, self.model.parameters(), create_graph=True)
            input_gradient_t = [torch.tensor(arr, dtype=torch.float32).to(self.device) for arr in input_gradient]        
            rec_loss = globals()[self.cost_fn]([gradient[0]], input_gradient_t)
            # reg loss
            for key, value in self.reg.items():
                if key == 'BN':
                    rec_loss += value * globals()[key](self.bn_layers, self.norm_inputs)
                else:
                    rec_loss += globals()[key](x_trial, value)
            rec_loss.backward()
            if 'signed' in self.config:
                x_trial.grad.sign()
            return rec_loss
        return closure

    def _run_trial(self, x_trial, input_data, labels):
        if self.reconstruct_label:
            output_test = self.model(x_trial)
            labels = torch.randn(output_test.shape[1]).to(self.device, self.dtype).requires_grad_(True)

            if self.config['optim'] == 'adam':
                optimizer = torch.optim.Adam([x_trial, labels], lr=self.config['lr'])
            elif self.config['optim'] == 'sgd':  # actually gd
                optimizer = torch.optim.SGD([x_trial, labels], lr=0.01, momentum=0.9, nesterov=True)
            elif self.config['optim'] == 'LBFGS':
                optimizer = torch.optim.LBFGS([x_trial, labels], lr=0.01)
            else:
                raise ValueError()
        else:
            if self.config['optim'] == 'adam':
                optimizer = torch.optim.Adam([x_trial], lr=self.config['lr'])
            elif self.config['optim'] == 'sgd':  # actually gd
                optimizer = torch.optim.SGD([x_trial], lr=0.01, momentum=0.9, nesterov=True)
            elif self.config['optim'] == 'LBFGS':
                optimizer = torch.optim.LBFGS([x_trial], lr=0.01)
            else:
                raise ValueError()

        num_trials = self.config['num_trials']

        if self.scheduler == 'MultiStepLR':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                             milestones=[num_trials // 2.667, num_trials // 1.6,

                                                                         num_trials // 1.142], gamma=0.1)   # 3/8 5/8 7/8
        elif self.scheduler == 'CosineAnnealingLR':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_trials, eta_min=0.0)
        start_time = time.time()
        try:
            for iteration in range(num_trials):
                x_trial.requires_grad = True
                closure = self._gradient_closure(optimizer, x_trial, input_data, labels)
                rec_loss = optimizer.step(closure)
                if self.scheduler:
                    scheduler.step()
                with torch.no_grad():
                    # Project into image space
                    x_trial.data = torch.clamp(x_trial, 0, 1).detach()
                    if 'print_interval' in self.config:
                        intv = self.config['print_interval']
                    else:
                        intv = 10
                    if (iteration + 1 == num_trials) or iteration == 1 or (iteration % (intv+1) == 0):
                        print(f'It: {iteration}. Rec. loss: {rec_loss.item():2.4f}.')
                        self.Server.logger.log(round=iteration, identity="Server", action="Reconstruction iter", message=f"It: {iteration}. Rec. loss: {rec_loss.item():2.4f}.")
                        if self.num_images == 1:
                            self.visual_image(x_trial, filename="res_"+str(iteration)+".png")
                        else:
                            x_show = torch.split(x_trial, 1)
                            label_show = torch.argmax(labels, -1).data.cpu().numpy().tolist()
                            self.visual_imglist(x_show, label_show,'result_img', filename="res_"+str(iteration)+".png")

        except KeyboardInterrupt:
            print('Trial procedure manually interruped.')
            pass
        print(f'Total time: {time.time()-start_time}.')
        self.Server.logger.log(round=num_trials, identity="Server", action="Time count", message=f"Recovery time per image {(time.time()-start_time)/self.num_images}")
        return x_trial.detach(), labels

    def reconstruction(self, shared_data, labels = None):
        shared_info=[]
        for i in range(len(shared_data["updates"])):
            uptensor = shared_data["parameters"][i] - shared_data["updates"][i]
            shared_info.append(uptensor)
        self.model = copy.deepcopy(self.Server.global_model).to(self.device)
        if 'BN' in self.reg:
            self.bn_layers, self.norm_inputs = collect_bn_layers(self.model)
        start_time = time.time()
        x = self._init_images(self.input_data_shape)
        # print("Initial x_trail: ", x)
        if labels == None:
            self.reconstruct_label = True
        else:
            # list to one-hot tensor
            # to the setting when only recovering part of imgs
            labels = labels[:self.num_images]
            labels = torch.LongTensor(labels)
            labels = F.one_hot(labels, num_classes = self.num_classes).float()
        # Recoving label using iDLG
        if self.reconstruct_label and self.num_images == 1 and self.config['label_inf'] == 'iDLG':
            last_weight_min = torch.argmin(torch.sum(shared_info[-2], dim=-1), dim=-1)
            labels = last_weight_min.detach().reshape((1,)).requires_grad_(False)
            print("Using iDLG to infer the label of img.\n Label: ", labels)
            self.reconstruct_label = False
        if self.reconstruct_label == True:
            def loss_fn(pred, labels):
                labels = torch.nn.functional.softmax(labels, dim=-1)
                return torch.mean(torch.sum(- labels * torch.nn.functional.log_softmax(pred, dim=-1), 1))
            self.loss_fn = loss_fn
        labels = labels.to(self.device, self.dtype)
        x_trial, labels = self._run_trial(x, shared_info, labels)
        print(f'Reconstructing time:{time.time()-start_time}s.')
        return x_trial, labels
      
