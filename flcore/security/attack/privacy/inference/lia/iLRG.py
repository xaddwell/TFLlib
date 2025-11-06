import torch
import copy
import numpy as np

from flcore.security.attack.privacy.inference.lia.base import BaseLabelInferenceAttack

# extend iDLG to a batch, following the iLRG
class iLRGAttack(BaseLabelInferenceAttack):
    def __init__(self, Server, config, device = 'cuda', logger=None):
        super().__init__('iLRG', Server, config)
        self.recover_num = None
        self.device = device
        if logger:
            self.logger = logger     
    
    def label_inference(self, shared_info, model, simplified=False, labels = None):
        self.model = copy.deepcopy(model).to(self.device)
        w_grad, b_grad = shared_info[-2], shared_info[-1]
        cls_rec_probs = self._get_rec_emb(w_grad, b_grad, self.model)
        # Recovered Labels
        rec_instances, mod_rec_instances = self._sim_iLRG(cls_rec_probs, b_grad, labels, self.gt_k) if simplified else self._iLRG(
            cls_rec_probs,
            b_grad,
            self.class_num,
            self.gt_k)

        rec_labels = labels if simplified else list(np.where(rec_instances > 0)[0])        
        #res = [rec_labels, rec_instances_nonzero, rec_instances, existences, mod_rec_instances]
        self.rec_labels = np.array(rec_labels)
        self.rec_instances = rec_instances
        if rec_labels == []:
            raise ValueError("label inferencing is failed!")
        return rec_instances, rec_labels
    
    # Recover Labels
    def _iLRG(self, probs, grad_b, n_classes, n_images):
        # Solve linear equations to recover labels
        coefs, values = [], []
        # Add the first equation: k1+k2+...+kc=K
        coefs.append([1 for _ in range(n_classes)])
        values.append(n_images)
        # Add the following equations
        for i in range(n_classes):
            coef = []
            for j in range(n_classes):
                if j != i:
                    coef.append(probs[j][i].item())
                else:
                    coef.append(probs[j][i].item() - 1)
            coefs.append(coef)
            values.append(n_images * grad_b[i])
        # Convert into numpy ndarray
        coefs = np.array(coefs)
        values = np.array(values)
        # Solve with Moore-Penrose pseudoinverse
        res_float = np.linalg.pinv(coefs).dot(values)
        # Filter negative values
        res = np.where(res_float > 0, res_float, 0)
        # Round values
        res = np.round(res).astype(int)
        res = np.where(res <= n_images, res, 0)
        err = res - res_float
        num_mod = np.sum(res) - n_images
        if num_mod > 0:
            inds = np.argsort(-err)
            mod_inds = inds[:num_mod]
            mod_res = res.copy()
            mod_res[mod_inds] -= 1
        elif num_mod < 0:
            inds = np.argsort(err)
            mod_inds = inds[:num_mod]
            mod_res = res.copy()
            mod_res[mod_inds] += 1
        else:
            mod_res = res

        return res, mod_res


    # Have Known about which labels exist
    def _sim_iLRG(self, probs, grad_b, exist_labels, n_images):
        # Solve linear equations to recover labels
        coefs, values = [], []
        # Add the first equation: k1+k2+...+kc=K
        coefs.append([1 for _ in range(len(exist_labels))])
        values.append(n_images)
        # Add the following equations
        for i in exist_labels:
            coef = []
            for j in exist_labels:
                if j != i:
                    coef.append(probs[j][i].item())
                else:
                    coef.append(probs[j][i].item() - 1)
            coefs.append(coef)
            values.append(n_images * grad_b[i])
        # Convert into numpy ndarray
        coefs = np.array(coefs)
        values = np.array(values)
        # Solve with Moore-Penrose pseudoinverse
        res_float = np.linalg.pinv(coefs).dot(values)
        # Filter negative values
        res = np.where(res_float > 0, res_float, 0)
        # Round values
        res = np.round(res).astype(int)
        res = np.where(res <= n_images, res, 0)
        err = res - res_float
        num_mod = np.sum(res) - n_images
        if num_mod > 0:
            inds = np.argsort(-err)
            mod_inds = inds[:num_mod]
            mod_res = res.copy()
            mod_res[mod_inds] -= 1
        elif num_mod < 0:
            inds = np.argsort(err)
            mod_inds = inds[:num_mod]
            mod_res = res.copy()
            mod_res[mod_inds] += 1
        else:
            mod_res = res

        return res, mod_res

    # Recover embeddings
    def _get_rec_emb(self, w_grad, b_grad, model, exp_thre=10, alpha=1):
        def find_last_linear_layer(model):
            """
            递归遍历模型，找到最后一个 nn.Linear 层
            :param model: nn.Module 类型的模型
            :return: 最后一个 nn.Linear 层
            """
            last_linear_layer = None
            
            # 遍历模型的子模块
            for layer in model.children():
                if isinstance(layer, torch.nn.Linear):
                    # 如果是 nn.Linear，更新最后一个线性层的引用
                    last_linear_layer = layer
                else:
                    # 如果是其他子模块，递归搜索
                    nested_layer = find_last_linear_layer(layer)
                    if nested_layer is not None:
                        last_linear_layer = nested_layer

            return last_linear_layer
    
        def get_emb(grad_w, grad_b, exp_thre=10):
            # Split scientific count notation
            sc_grad_b = '%e' % grad_b
            sc_grad_w = ['%e' % w for w in grad_w]
            real_b, exp_b = float(sc_grad_b.split('e')[0]), int(sc_grad_b.split('e')[1])
            real_w, exp_w = np.array([float(sc_w.split('e')[0]) for sc_w in sc_grad_w]), \
                            np.array([int(sc_w.split('e')[1]) for sc_w in sc_grad_w])
            # Deal with 0 case
            if real_b == 0.:
                real_b = 1
                exp_b = -64
            # Deal with exponent value
            exp = exp_w - exp_b
            exp = np.where(exp > exp_thre, exp_thre, exp)
            exp = np.where(exp < -1 * exp_thre, -1 * exp_thre, exp)

            def get_exp(x):
                return 10 ** x if x >= 0 else 1. / 10 ** (-x)

            exp = np.array(list(map(get_exp, exp)))
            # Calculate recovered average embeddings for batch_i (samples of class i)
            res = (1. / real_b) * real_w * exp
            res = torch.from_numpy(res).to(torch.float32)
            return res

        def post_process_emb(embedding, model, device, alpha=0.01):
            embedding = embedding.to(device)
            # Feed embedding into FC-Layer to get probabilities.
            linear = find_last_linear_layer(model)
            out = linear(embedding) * alpha
            prob = torch.softmax(out, dim=-1)
            return prob
        
        cls_rec_probs = []
        for i in range(self.class_num):
            # Recover class-specific embeddings and probabilities
            cls_rec_emb = get_emb(w_grad[i], b_grad[i])
            # if (not args.silu) and (not args.leaky_relu):
            #     cls_rec_emb = torch.where(cls_rec_emb < 0., torch.full_like(cls_rec_emb, 0), cls_rec_emb)
            # cls_rec_emb = torch.where(w_grad[i] < 0., torch.full_like(w_grad[i], 0), w_grad[i])
            cls_rec_prob = post_process_emb(embedding=cls_rec_emb,
                                            model=model,
                                            device=self.device,
                                            alpha=alpha) # 'Factor for scaling outputs'
            cls_rec_probs.append(cls_rec_prob)       
        return cls_rec_probs
        
