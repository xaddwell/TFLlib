from abc import ABC
import numpy as np
import yaml
import random
import torch
from flcore.security.attack.privacy.inference.mia.utils import MIA_metrics

class MembershipInferenceAttack(ABC):
    
    def __init__(self, attack_name, Server, conf_path=None, conf2=None, c_model=None):
        print("initialzing membership inf attack")
        self.attack_name = attack_name
        self.classifier = c_model
        self.config = None
        self.cid = None
        self.Server = Server
        self.num_classes = Server.args.num_classes
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
        if conf_path:
            with open(conf_path, 'r', encoding='utf-8') as f:
                self.config = yaml.load(f.read(), Loader=yaml.FullLoader)
                self.cid = self.config['cid']
            if conf2:
                # merge a configparse and a yaml_dict
                self.config = {**self.config, **vars(conf2)}#self.merge_config(self.config, conf2)
        else:
            self.config = vars(conf2)
        if 'attack_mode' not in self.config:
            self.config['attack_mode'] = 'Passive'
        if 'cid' in conf2:
            self.cid = conf2.cid
        if self.cid is None:
            self.cid = random.randint(0, self.config['num_clients']-1)
            print("The random choosed victim client is ", self.cid)
        self.config['atk_round'] = -1
        self.batch_size=self.config['batch_size']
        self.config["attack_type"] = 'MIA'

    def merge_config(self, conf1, conf2):
        # Convert configparser to a dictionary recursively
        # Priority: attacker.yaml is lower than the training config
        config_dict = {section: dict(conf2.items(section)) for section in conf2.sections()}
        for key in config_dict:
            if key in conf1:
                if isinstance(conf1[key], dict) and isinstance(config_dict[key], dict):
                    self.merge_config(conf1[key], config_dict[key])
            else:
                conf1[key] = config_dict[key]   
        return conf1
    
    def preprocess_train_data(self, nonmem_data, mem_data):
        len_data = min(len(mem_data), len(nonmem_data))
        mem_length = len(mem_data) // 2
        nonmem_length = len(nonmem_data) // 2

        print("length of mem data:{}, length of nonmember data:{}".format(mem_length, nonmem_length))
        mem_train, mem_test = torch.utils.data.random_split(mem_data, [len_data, 
                                                                        len(mem_data) - (len_data * 1)])

        nonmem_train, nonmem_test = torch.utils.data.random_split(nonmem_data, [len_data,
                                                                                len(nonmem_data) - (len_data * 1)])

        mem_train, mem_test, nonmem_train, nonmem_test = list(mem_train), list(mem_test), list(nonmem_train), list(
            nonmem_test)
        print(f"length of train data: {len(mem_train)+len(nonmem_train)}, length of test data:{len(mem_test)+len(nonmem_test)}")

        for i in range(len(mem_train)):
            mem_train[i] = mem_train[i] + (1,)
        for i in range(len(nonmem_train)):
            nonmem_train[i] = nonmem_train[i] + (0,)
        for i in range(len(nonmem_test)):
            nonmem_test[i] = nonmem_test[i] + (0,)
        for i in range(len(mem_test)):
            mem_test[i] = mem_test[i] + (1,)

        attack_train = mem_train + nonmem_train
        attack_test = mem_test + nonmem_test

        batch_size=self.config['train']['batch_size']
        self.attack_train_loader = torch.utils.data.DataLoader(
            attack_train, batch_size=batch_size, shuffle=True, num_workers=2)
        self.attack_test_loader = torch.utils.data.DataLoader(
            attack_test, batch_size=batch_size, shuffle=True, num_workers=2)
        
        return

    def preprocess_shadow_data(self, aux_data, test_mem_data, test_non_mem_data):
        data_length = len(aux_data) // 2
        shadow_train, shadow_test= torch.utils.data.random_split(aux_data, [data_length, len(aux_data) - data_length])

        mem_train, nonmem_train = list(shadow_train), list(shadow_test)
        mem_test, nonmem_test = list(test_mem_data), list(test_non_mem_data)

        for i in range(len(mem_train)):
            mem_train[i] = mem_train[i] + (1,)
        for i in range(len(nonmem_train)):
            nonmem_train[i] = nonmem_train[i] + (0,)

        for i in range(len(mem_test)):
            mem_test[i] = mem_test[i] + (1,)
        for i in range(len(nonmem_test)):
            nonmem_test[i] = nonmem_test[i] + (0,)

        train_length = min(len(mem_train), len(nonmem_train))
        test_length = min(len(mem_test), len(nonmem_test))

        mem_train, _ = torch.utils.data.random_split(mem_train, [train_length, len(mem_train) - train_length])
        non_mem_train, _ = torch.utils.data.random_split(nonmem_train, [train_length, len(nonmem_train) - train_length])
        mem_test, _ = torch.utils.data.random_split(mem_test, [test_length, len(mem_test) - test_length])
        non_mem_test, _ = torch.utils.data.random_split(nonmem_test, [test_length, len(nonmem_test) - test_length])

        attack_train = mem_train + non_mem_train
        attack_test = mem_test + non_mem_test
        
        self.attack_train_loader = torch.utils.data.DataLoader(
            attack_train, batch_size=self.batch_size, shuffle=True, num_workers=2)
        self.attack_test_loader = torch.utils.data.DataLoader(
            attack_test, batch_size=self.batch_size, shuffle=True, num_workers=2)
        self.shadow_test_loader = torch.utils.data.DataLoader(
            mem_train, batch_size=self.batch_size, shuffle=True, num_workers=2)
        self.shadow_train_loader = torch.utils.data.DataLoader(
            non_mem_train, batch_size=self.batch_size, shuffle=True, num_workers=2)
        print("For training shadow data: length of mem data:{}, length of nonmember data:{}".format(len(mem_train), len(mem_test)))
        print("ratio of mem:nonmen in training stage is:{}, in testing stage is:{}".format(len(mem_train)/len(nonmem_train), len(mem_test)/len(nonmem_test)))
        return
    
    def train_shadow_models(self):
        raise NotImplementedError("Shadow model training process not completed")

    def train_classifier(self):
        raise NotImplementedError("Classifier training process not completed")
    
    def MIA(self, target):
        raise NotImplementedError("MIA attack not completed")
    
    def saveModel(self, model, path):
        torch.save(model.state_dict(), path)

    def metrics(self, y_pred, labels, atk_labels):
        checker = MIA_metrics(y_pred, labels, atk_labels)
        metric_dict = checker.run(self.config['metrics'])
        return metric_dict