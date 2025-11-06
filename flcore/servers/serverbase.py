

import torch
import os
import numpy as np
import h5py
import copy
import time
import random
import torch.utils
import torch.utils.data
from tqdm import tqdm
import concurrent.futures
from sklearn import metrics
from sklearn.preprocessing import label_binarize

from utils.info import write_info
from utils import get_best_gpu, LoggerBase
from flcore.simulation import comm_hetero, dev_hetero
from flcore.security.attack import initialize_server_attacker

class Server(object):
    def __init__(self, args, model, train_data, test_data, times):
        # Set up the main attributes
        self.args = args
        self.timeout = self.args.timeout
        self.dataset = args.data_name
        self.num_classes = args.num_classes
        self.global_rounds = args.global_rounds
        self.local_epochs = args.local_epochs
        self.batch_size = args.batch_size
        self.global_model = copy.deepcopy(model)
        self.train_data = train_data
        test_data.set_tokenizer(model.tokenizer)
        self.test_data = test_data
        self.num_clients = args.num_clients
        self.join_ratio = args.join_ratio
        self.random_join_ratio = args.random_join_ratio
        self.num_join_clients = int(self.num_clients * self.join_ratio)
        self.current_num_join_clients = self.num_join_clients
        self.algorithm = args.algorithm
        self.task_id = args.task_id

        self.save_folder_name = os.path.join("./log", self.args.save_folder_name)
        self.args.save_folder_name = self.save_folder_name
        if not os.path.exists(self.save_folder_name):
            os.makedirs(self.save_folder_name)
        self.logger = LoggerBase(
            log_name="log.log", 
            log_file_path=self.save_folder_name
            )

        self.clients = []
        self.selected_clients = []
        self.train_slow_clients = []
        self.send_slow_clients = []

        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []

        self.rs_test_acc = []
        self.rs_test_auc = []
        self.rs_train_loss = []

        
        self.times = times
        self.eval_gap = args.eval_gap
        self.poison_eval = None
        self.attack_flag = False
        self._current_round = args.start_round
        self._start_round = args.start_round
        

    def init_attacker(self):
        self.attack_flag = True
        self.attacker = initialize_server_attacker(self, self.args)


    def set_clients(self, clientObj):
        
        # Enforce system heterogeneity of clients.
        network_condition = [comm_hetero.assign_network_condition(self.args.comm_hetero) \
                             for _ in range(self.args.num_clients)]
        device_condition = [dev_hetero.assign_device_condition(self.args.dev_hetero) \
                            for _ in range(self.args.num_clients)]

        data_ratio = np.array([len(self.train_data[i][0]) for i in range(self.args.num_clients)])

        data_ratio = data_ratio / np.mean(data_ratio)

        bd_attackers_set = random.sample(range(self.args.num_clients), self.args.bd_clients)
        remained_clients = list(set(range(self.args.num_clients)) - set(bd_attackers_set))
        bzt_attackers_set = random.sample(remained_clients, self.args.bzt_clients)
        self.args.bd_client_ids = bd_attackers_set
        self.args.bzt_client_ids = bzt_attackers_set

        self.logger.log(round=self._current_round, identity="Server", action="Initialize", message=f"BKD attackers-{bd_attackers_set}")
        self.logger.log(round=self._current_round, identity="Server", action="Initialize", message=f"BZT attackers-{bzt_attackers_set}")
        

        def __create_client(identifier, datasets):
            if self._malicious_global_models:
                set_model = self._malicious_global_models[identifier]
            else:
                set_model = self.global_model
            client = clientObj(args=self.args, id=identifier, 
                               model=set_model,
                               train_data=datasets[0], 
                               test_data=datasets[1],
                               data_ratio=data_ratio[identifier],
                               network_condition=network_condition[identifier],
                               device_condition=device_condition[identifier])
            return client
        
        self._malicious_global_models = None
        if 'privacy_attack' in self.args and self.args.privacy_attack is not None:
            self.init_attacker()        
            if self.attacker.config['attack_mode'] == 'Active':
                self._malicious_global_models = self.attacker.place_malicious_block(self.global_model, self.num_clients)

        jobs = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(int(self.args.num_workers), os.cpu_count() - 1)) as workhorse:
            for identifier in range(self.num_clients):
                jobs.append(workhorse.submit(__create_client, identifier, self.train_data[identifier]))

            for task in concurrent.futures.as_completed(jobs):
                self.clients.append(task.result())
        
        write_info(self.args)
            
    
    def select_clients(self):
    
        if self.random_join_ratio:
            self.current_num_join_clients = np.random.choice(range(self.num_join_clients, self.num_clients+1), 1, replace=False)[0]
        else:
            self.current_num_join_clients = self.num_join_clients

        fixed_attackers = []
        if self.args.bd_fixed:
            fixed_attackers = fixed_attackers + self.args.bd_client_ids
        if self.args.bzt_fixed:
            fixed_attackers = fixed_attackers + self.args.bzt_client_ids
        if self.args.cid >= 0 and self.attack_flag:
            fixed_attackers.append(self.args.cid)
        '''
        if self._current_round < self.args.bd_start_round or self._current_round > self.args.bd_end_round:
            remained_clients = [client for client in self.clients if client.id not in fixed_attackers]
            selected_clients = list(np.random.choice(remained_clients, self.current_num_join_clients,replace=False))
        '''
        if len(fixed_attackers) > 0:
            selected_clients = [client for client in self.clients if client.id in fixed_attackers]
            remained_clients = [client for client in self.clients if client.id not in fixed_attackers]
            selected_clients = selected_clients + \
                list(np.random.choice(remained_clients, 
                                      self.current_num_join_clients - len(selected_clients), 
                                      replace=False))
        else:
            selected_clients = list(np.random.choice(self.clients, self.current_num_join_clients, replace=False))
        
        self.logger.log(round=self._current_round, identity="Server", action="Selection", message=f"SelectedClients-{[client.id for client in selected_clients]}")
        
        return selected_clients


    def send_models(self):
        assert (len(self.clients) > 0)
        for i, client in enumerate(self.clients):
            start_time = time.time()
            client.set_parameters(self.global_model)
            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['current_cost'] = 2 * (time.time() - start_time)
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)


    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        train_time_cost = []
        slow_clients = []
        total_time_cost = 0
        tot_samples = 0

        for client in self.selected_clients:
            client_time_cost = client.train_time_cost['current_cost'] + \
                client.send_time_cost['current_cost']
            train_time_cost.append(client_time_cost)
            total_time_cost += client_time_cost
        avg_time_cost = total_time_cost / len(self.selected_clients)
        avg_time_cost_list = [t/avg_time_cost for t in train_time_cost]

        for idx, client in enumerate(self.selected_clients):

            ############### communication heterogeneity ###############
            if self.args.comm_hetero != 1:
                # if client.network_condition < random.random()*avg_time_cost_list[idx]:
                if client.network_condition < random.random():
                    slow_clients.append((client.id, client.network_condition))
                    continue
            ###########################################################

            tot_samples += client.train_samples
            self.uploaded_ids.append(client.id)
            self.uploaded_weights.append(client.train_samples)
            self.uploaded_models.append(client.model)

        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples
        
        if len(slow_clients) > 0:
            # self.logger.log(round=self._current_round, identity="Server", action="Summary", message=f"CostTime-{train_time_cost}")
            self.logger.log(round=self._current_round, identity="Server", action="Summary", message=f"SlowClients-{slow_clients}")


    def aggregate_parameters(self):
        assert (len(self.uploaded_models) > 0)

        if self.attack_flag and self._current_round >= (self.attacker.config['atk_round'] + self._start_round) and self.attacker.config['atk_round'] >= 0:
            # target client may not be chosen
            if self.attack_aggregation():
                self.attack_flag = False

        self._current_round += 1
        self.global_model = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_model.parameters():
            param.data.zero_()
        
        # self.global_model.cpu()
        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            self.add_parameters(w, client_model)


    def attack_aggregation(self):
        target_client = None
        for client in self.clients:
            if client.id == self.attacker.cid:
                target_client = client
                break

        if target_client == None or self.attacker.cid not in self.uploaded_ids:
            return False

        # attack the aggreation phrase
        self.logger.log(round=self._current_round, identity="Server", action="Note", message="Begin to attack in aggregation phrase...")

        global_params = []
        target_params = []
        target_updates = []

        # The capability of server attacker in FedAvg
        # the update of target client
        self.last_round_ckpt = self.global_model.parameters()
        # the global params of last round

        self.current_round_ckpt = self.uploaded_models[self.uploaded_ids.index(self.attacker.cid)].parameters()

        for target_t, global_t in zip(self.current_round_ckpt, self.last_round_ckpt):
            target_params.append(target_t.clone().detach().cpu())
            up_torch = (global_t.cpu() - target_t.cpu()).clone().detach()
            target_updates.append(up_torch)
            global_params.append(global_t.clone().detach().cpu())

        tr_img, tr_label = target_client.get_ground_truth()

        if self.attacker.config["attack_type"] == 'inversion':
            self.attacker.visual_imglist(tr_img, tr_label, os.path.join(self.save_folder_name,"gt_img"), filename="tr_img.png")
            share_data = dict(updates=target_params, parameters=global_params)
            res_img, label_show = self.attacker.reconstruction(share_data, labels = tr_label) 
            x_show = torch.split(res_img, 1)
            self.attacker.visual_imglist(x_show, tr_label, os.path.join(self.save_folder_name, 'result_img'), 'res_img.png')
            metrics_dict = self.attacker.metrics(res_img.cuda(), torch.stack(tr_img)[:res_img.size(0)].cuda(), net='resnet')

        elif self.attacker.config["attack_type"] == 'label_inf':
            self.logger.log(round=self._current_round, identity="Server", action="Attack", message=f"Ground truth labels are: {tr_label[:self.batch_size]}")
            print(f"Ground truth labels are: {tr_label[:self.batch_size]}")
            if self.attacker.attack_name == 'iLRG':
                recover_num, label_show = self.attacker.label_inference([t.numpy() for t in target_updates], self.global_model)
            else:
                recover_num, label_show = self.attacker.label_inference([t.numpy() for t in target_updates])
            #label_show.sort()
            self.logger.log(round=self._current_round, identity="Server", action="Attack", message=f"Recovered number of batch is:{recover_num} The reference value of this batch:{label_show}")
            print("Recovered number of batch is:{} The reference value of this batch:{}".format(recover_num, label_show))
            self.attacker.metrics(tr_label[:self.batch_size])
        return True


    def add_parameters(self, w, client_model):
        update = client_model.state_dict()
        if self.args.clip_norm:
            clip_updates = {}
            global_params = self.global_model.state_dict()
            clip_factor = self.args.clip_factor if self.args.clip_factor > 0 else 1
            for key, var in update.items():
                if 'num_batches_tracked' in key or 'running_mean' in key or 'running_var' in key:
                    clip_updates[key] = var
                    continue
                diff = var - global_params[key]
                l2_update = torch.norm(diff, p=2)
                new_update = diff.div_(max(1, l2_update/clip_factor))
                clip_updates[key] = global_params[key] + new_update
            update = clip_updates
        for key, server_param in self.global_model.named_parameters():
            server_param.data += update[key].data.clone() * w

        # for server_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
        #     server_param.data += client_param.data.clone() * w


    def save_global_model(self):
        
        if self.attack_flag == True and self.attacker.config['atk_round'] == -1:
            self.attack_model()
        model_path = os.path.join("/home/zzm/RFLlib/pretrain", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_name = f"server_{self._current_round+1}.pt"
        
        torch.save(self.global_model.state_dict(), 
                   os.path.join(self.save_folder_name, model_name))


    def attack_model(self):
        target_client = None
        for client in self.clients:
            if client.id == self.attacker.cid:
                target_client = client
                break
        tr_data = target_client.train_data
        pred,c,y = self.attacker.MIA(tr_data)
        metric = self.attacker.metrics(pred, c, y)
        print(metric)

    
    def load_testloader(self):
        if self.args.eval_type == "global":
            return torch.utils.data.DataLoader(
                self.test_data, 
                batch_size=self.args.batch_size,
                shuffle=True
            )
        else:
            # aggregate the test data of all clients
            test_data_target = []
            test_data_source = []
            for client in self.clients:
                for _, test_data in enumerate(torch.utils.data.DataLoader(client.test_data, batch_size=len(client.test_data))):
                    test_data_source.append(test_data[0])
                    test_data_target.append(test_data[1])
            
            dataset = torch.utils.data.TensorDataset(torch.cat(test_data_source), torch.cat(test_data_target))
            
            return torch.utils.data.DataLoader(
                dataset,
                batch_size=self.args.batch_size,
                shuffle=True
            )
        
    def save_results(self):
        if (len(self.rs_test_acc)):
            file_path = self.save_folder_name + "/results.h5"
            with h5py.File(file_path, 'w') as hf:
                hf.create_dataset('rs_test_acc', data=self.rs_test_acc)
                hf.create_dataset('rs_test_auc', data=self.rs_test_auc)
                hf.create_dataset('rs_train_loss', data=self.rs_train_loss)
    

    def test_metrics(self):

        num_samples = []
        tot_correct = []
        tot_auc = []
        jobs = []
        ids = []

        def __evaluate_clients(client):
            ct, ns, auc = client.test_metrics(self._current_round)
            tot_correct.append(ct*1.0)
            tot_auc.append(auc*ns)
            num_samples.append(ns)
            ids.append(client.id)
            # return ct, ns, auc, client.id
        
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(
                int(self.args.num_workers), 
                os.cpu_count() - 1)) as workhorse:
            
            for client in self.clients:
                jobs.append(workhorse.submit(__evaluate_clients, client))

            for task in concurrent.futures.as_completed(jobs):
                task.result(timeout=self.timeout)

        return ids, num_samples, tot_correct, tot_auc


    def train_metrics(self):
        num_samples = []
        losses = []
        jobs = []
        ids = []
        def __evaluate_clients(client):
            cl, ns = client.train_metrics()
            return cl, ns, client.id

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(
                int(self.args.num_workers),
                os.cpu_count() - 1)) as workhorse:
            
            for client in self.clients:
                jobs.append(workhorse.submit(__evaluate_clients, client))
            for task in concurrent.futures.as_completed(jobs):
                cl, ns, client_id = task.result(timeout=self.timeout)
                num_samples.append(ns)
                losses.append(cl*1.0)
                ids.append(client_id)

        return ids, num_samples, losses

    
    def evaluate_local(self, acc=None, loss=None):
        self.logger.log(round=self._current_round, identity="Server", action="Note", message="Start evaluate on all clients...")
        
        stats = self.test_metrics()
        stats_train = self.train_metrics()

        test_acc = sum(stats[2])*1.0 / sum(stats[1])
        test_auc = sum(stats[3])*1.0 / sum(stats[1])
        train_loss = sum(stats_train[2])*1.0 / sum(stats_train[1])
        accs = [a / n for a, n in zip(stats[2], stats[1])]
        aucs = [a / n for a, n in zip(stats[3], stats[1])]
        

        if acc == None:
            self.rs_test_acc.append(test_acc)
        else:
            acc.append(test_acc)
        
        if loss == None:
            self.rs_train_loss.append(train_loss)
        else:
            loss.append(train_loss)

        
        self.logger.log(round=self._current_round, identity="Server", action="Local Evaluate", 
                        message=f"TrainLoss-{train_loss:.4f}, TestAccurancy-{test_acc:.4f}, AccStd-{np.std(accs):.4f}, TestAUC-{test_auc:.4f}, AUCStd-{np.std(aucs):.4f}")


    def evaluate_global(self):
        self.logger.log(round=self._current_round, identity="Server", 
                        action="Note", message="Start global test ...")

        loss = torch.nn.CrossEntropyLoss()
        loader = self.load_testloader()
        device = get_best_gpu()
        self.global_model.eval().to(device)
        correct, total, total_loss = 0, 0, 0
        y_prob, y_true = [], []
        label_list = [0] * self.num_classes

        with torch.no_grad():
            for batch in loader:
                if len(batch) == 2:
                    x, y = batch
                    x, y = x.to(device), y.to(device)
                    output = self.global_model(x)
                else:
                    x, mask, y = batch
                    x, mask, y = x.to(device), mask.to(device), y.to(device)
                    output = self.global_model(x, mask).logits
                
                if y.dim() == 2:
                    y = y.squeeze(1)
                
                total_loss += loss(output, y).item() * y.size(0)
                pred = output.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
                y_prob.append(output.detach().cpu().numpy())
                y_true.append(label_binarize(y.detach().cpu().numpy(), classes=np.arange(output.size(1))))

                # for i in range(len(y)):
                #     label_list[y[i].item()] += 1

        # self.logger.log(round=self._current_round, identity="Server", action="Note", message=f"Label distribution: {label_list}")
                
            
        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        if self.num_classes == 2:
            y_prob = y_prob[:, 1]
            auc = metrics.roc_auc_score(y_true, y_prob)
        else:
            auc = metrics.roc_auc_score(y_true, y_prob, average='micro')

        self.logger.log(round=self._current_round, identity="Server", action="Global Evaluate", message=f"TestAccuracy: {correct / total:.4f} TestLoss: {total_loss / total:.4f} TestAUC: {auc:.4f}")


