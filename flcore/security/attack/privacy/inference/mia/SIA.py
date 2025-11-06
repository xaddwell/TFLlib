
import copy
import numpy as np
import torch
import torch.nn as nn
from flcore.security.attack.privacy.inference.mia.base import MembershipInferenceAttack


class SIAAttacker(MembershipInferenceAttack):
    def __init__(self, Server, conf2=None):
        conf_path = None
        super().__init__("SIA", Server, conf_path, conf2)
        self.dict_mia_users = Server.num_join_clients
    
    def MIA(self, nonuse):
        correct_loss = 0
        len_set = 0
        for idx in range(self.dict_mia_users):

            y_loss_all = []
            dataset_local = self.Server.clients[idx].train_data
            dataset_local = torch.utils.data.DataLoader(dataset_local, batch_size=self.config['batch_size'], shuffle=True, num_workers=2)
            # evaluate each party's training data on each party's model
            for local in range(self.dict_mia_users):

                y_losse = []

                idx_tensor = torch.tensor(idx)
                net = copy.deepcopy(self.Server.global_model).to(self.device)
                net.eval()
                for id, (data, target) in enumerate(dataset_local):
                    data, target = data.to(self.device), target.to(self.device)
                    idx_tensor = idx_tensor.to(self.device)
                    log_prob = net(data)
                    # prediction loss based attack: get the prediction loss of the test sample
                    loss = nn.CrossEntropyLoss(reduction='none')
                    y_loss = loss(log_prob, target)
                    y_losse.append(y_loss.cpu().detach().numpy())

                y_losse = np.concatenate(y_losse).reshape(-1)
                y_loss_all.append(y_losse)

            y_loss_all = torch.tensor(y_loss_all).to(self.device)

            # test if the owner party has the largest prediction probability
            # get the parties' index of the largest probability of each sample
            index_of_party_loss = y_loss_all.min(0, keepdim=True)[1]
            correct_local_loss = index_of_party_loss.eq(
                idx_tensor.repeat_interleave(len(dataset_local.dataset))).long().cpu().sum()

            correct_loss += correct_local_loss
            len_set += len(dataset_local.dataset)

        # calculate membership inference attack accuracy
        accuracy_loss = 100.00 * correct_loss / len_set

        print('\nTotal attack accuracy of prediction loss based attack: {}/{} ({:.2f}%)\n'.format(correct_loss, len_set,
                                                                                                  accuracy_loss))

        return accuracy_loss, None, None
    
