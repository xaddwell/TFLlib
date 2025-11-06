'''
Description: 
Version: 1.0
Author: Jiahao Chen
Date: 2024-06-29 11:26:14
LastEditors: Jiahao Chen
LastEditTime: 2024-07-01 20:15:06
'''

from flcore.servers.serveravg import FedAvg
from flcore.servers.serverprox import FedProx
from flcore.servers.serverdyn import FedDyn
from flcore.servers.servermoon import MOON
from flcore.servers.servergen import FedGen
from flcore.servers.serverscaffold import SCAFFOLD
from flcore.servers.serverntd import FedNTD

from flcore.models import BaseHeadSplit

import copy
import torch.nn as nn

def server_selector(args, model, train_data, test_data, curr_time):
    # select algorithm
    if args.algorithm == "FedAvg":
        server = FedAvg(args, model, train_data, test_data, curr_time)

    elif args.algorithm == "FedProx":
        server = FedProx(args, curr_time)

    elif args.algorithm == "FedDyn":
        server = FedDyn(args, curr_time)

    elif args.algorithm == "MOON":
        args.head = copy.deepcopy(args.model.fc)
        args.model.fc = nn.Identity()
        args.model = BaseHeadSplit(args.model, args.head)
        server = MOON(args, curr_time)

    elif args.algorithm == "FedGen":
        args.head = copy.deepcopy(args.model.fc)
        args.model.fc = nn.Identity()
        args.model = BaseHeadSplit(args.model, args.head)
        server = FedGen(args, curr_time)

    elif args.algorithm == "SCAFFOLD":
        server = SCAFFOLD(args, curr_time)

    elif args.algorithm == "FedNTD":
        server = FedNTD(args, curr_time)

    else:
        raise NotImplementedError
    
    return server