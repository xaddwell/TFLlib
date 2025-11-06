'''
Description: 
Version: 1.0
Author: Jiahao Chen
Date: 2024-06-27 22:04:25
LastEditors: Jiahao Chen
LastEditTime: 2024-07-20 20:16:13
'''
#!/usr/bin/env python

import copy
import torch
import argparse
import os
import time
import warnings
import numpy as np
import torchvision
import logging
import random

from flcore.fedatasets.data import construct_datasets
from flcore.servers.server_route import server_selector
from flcore.models import load_model
from flcore.config import parse_args

from flcore.utils import get_best_gpu, set_random_seed
from flcore.utils.result_utils import average_data
from flcore.utils.mem_utils import MemReporter


warnings.simplefilter("ignore")


def run(args):

    time_list = []
    reporter = MemReporter()
    start = time.time()
    train_data, test_data, args = construct_datasets(args)
    model, args = load_model(args)
    args.save_folder_name = f"{args.exp_name}/{args.task_id}"
    server = server_selector(args, model, train_data, test_data, start)
    server.train()
    time_list.append(time.time()-start)
    print(f"\nAverage time cost: {round(np.average(time_list), 2)}s.")
    print("All done!")
    reporter.report()


if __name__ == "__main__":
    total_start = time.time()
    args = parse_args()
    args.device =get_best_gpu()
    set_random_seed(args.seed)
    print("=" * 50)
    run(args)