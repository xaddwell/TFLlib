'''
Description: 
Version: 1.0
Author: Jiahao Chen
Date: 2024-06-27 22:04:25
LastEditors: Jiahao Chen
LastEditTime: 2024-07-07 22:26:11
'''

import os
import time
from tqdm import tqdm
import concurrent.futures

from flcore.clients.clientavg import clientAVG
from flcore.servers.serverbase import Server



class FedAvg(Server):
    def __init__(self, args, model, train_data, test_data, times):
        super().__init__(args, model, train_data, test_data, times)
        # select slow clients
        self.set_clients(clientAVG)

        self.logger.log(round=self._current_round, identity="Server", action="Initialize", 
                        message=f"Join ratio / total clients: {self.join_ratio} / {self.num_clients}")
        self.logger.log(round=self._current_round, identity="Server", action="Initialize", 
                        message=f"Finished creating server and clients.")

        # self.load_model()
        self.Budget = []


    def train(self):
        
        def __update_clients(client, round_id):
            l,c = client.train(round_id)
            for ll, cc in zip(l, c):
                lc[ll] += cc
        
        for i in tqdm(range(self._start_round, self.global_rounds+self._start_round+1)):
            lc = [0 for i in range(self.args.num_classes)]
            s_t = time.time()
            self.selected_clients = self.select_clients()

            jobs = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(int(self.args.num_workers), os.cpu_count() - 1)) as workhorse:
                for client in self.selected_clients:
                    jobs.append(workhorse.submit(__update_clients, client, i))
                self.logger.log(round=i, identity="Server", action="Note", message=f"Start training")
                for task in concurrent.futures.as_completed(jobs):
                    tmp = task.result(timeout=self.timeout)
            
            # print(f"Assumed Labels: ")
            # for i, c in enumerate(lc):
            #     print(f"L{i}: {c} ", end="")
            # print("Received models")

            self.receive_models()
            self.aggregate_parameters()
            self.send_models()
            
            if (i+1)%self.eval_gap == 0:
                
                self.logger.log(round=i, identity="Server", action="Note", message=f"Evaluate global model")
                
                if self.args.eval_type == "local":
                    self.evaluate_local()
                elif self.args.eval_type == "global":
                    self.evaluate_global()
                
                if self.poison_eval == None and self.args.bd_clients > 0:
                    for client in self.clients:
                        if client.id in self.args.bd_client_ids:
                            self.poison_eval = client.attacker.poison_eval
                            break
                elif self.args.bd_clients > 0:
                    self.poison_eval(self.global_model, self.load_testloader())
            

            self.Budget.append(time.time() - s_t)
            self.logger.log(round=i, identity="Server", action="Summary", message=f"Total Time cost-{self.Budget[-1]:.4f}")

            if (i+1) % self.args.save_interval == 0:
                self.save_global_model()

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        if self.rs_test_auc:
            print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()
