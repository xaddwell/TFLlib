

import importlib
import json
import logging
import os


import os
import gc
import torch
import logging
import random
import concurrent.futures

from flcore.fedatasets.basedataset import BaseDataset
from flcore.fedatasets.other.tinyimagenet import fetch_tinyimagenet
from flcore.fedatasets.other.cinic10 import fetch_cinic10
from flcore.fedatasets.other.cifar10 import fetch_cifar10
from flcore.fedatasets.other.cifar100 import fetch_cifar100
from flcore.fedatasets.other.femnist import fetch_femnist
from flcore.fedatasets.other.adult import fetch_adult
from flcore.fedatasets.other.heart import fetch_heart
from flcore.fedatasets.other.gleam import fetch_gleam
from flcore.fedatasets.other.ucihar import fetch_ucihar
from flcore.fedatasets.other.texas100 import fetch_texas100
from flcore.fedatasets.other.purchase100 import fetch_purchase100
from flcore.fedatasets.other.creditcard import fetch_creditcard
from flcore.fedatasets.other.sent140 import fetch_sent140
from flcore.fedatasets.other.imdb import fetch_imdb
from flcore.fedatasets.other.agnews import fetch_agnews


from flcore.fedatasets.utils.split import simulate_split
from flcore.fedatasets.utils import TqdmToLogger, stratified_split

from flcore.fedatasets import dataset_dict, dataset_root



def construct_datasets(args):
    """Fetch and split requested datasets.
    """
    dataset_name = args.data_name
    data_path = os.path.join(dataset_root,dataset_name.lower())
    dataset_args = dataset_dict[dataset_name]
    num_classes = dataset_args["num_classes"]
    dataset_args["root"] = data_path
    dataset_args["name"] = dataset_name
    args.num_classes = num_classes
    args.domain = dataset_args["domain"]
    test_size = args.test_size

    # error manager
    def _check_and_raise_error(entered, targeted, msg, eq=True):
        if eq:
            if entered == targeted: # raise error if eq(==) condition meets
                err = f'[{dataset_name.upper()}] `{entered}` {msg} is not supported for this dataset!'
                print(err)
                raise AssertionError(err)
        else:
            if entered != targeted: # raise error if neq(!=) condition meets
                err = f'[{dataset_name.upper()}] `{targeted}` {msg} is only supported for this dataset!'
                print(err)
                raise AssertionError(err)
    
    raw_train, raw_test = BaseDataset(), BaseDataset()
    split_map, client_datasets = None, None
          
    ##################### fetch dataset #########################
    if dataset_name == 'CelebA':
        _check_and_raise_error(args.split_type, 'pre', 'split scenario', False)
        _check_and_raise_error(args.eval_type, 'local', 'evaluation type', False)
        raw_train, raw_test, args = \
            fetch_femnist(args, dataset_args=dataset_args, root=data_path)
    
    elif dataset_name == 'FEMNIST':
        _check_and_raise_error(args.split_type, 'pre', 'split scenario', False)
        _check_and_raise_error(args.eval_type, 'local', 'evaluation type', False)
        client_datasets = fetch_femnist(args, root=data_path, test_size=test_size)

    elif dataset_name == 'CIFAR10':
        _check_and_raise_error(args.split_type, 'pre', 'split scenario')
        raw_train, raw_test, create_instance = \
            fetch_cifar10(args=args,root=data_path, test_size=test_size)
    
    elif dataset_name == 'CIFAR100':
        _check_and_raise_error(args.split_type, 'pre', 'split scenario')
        raw_train, raw_test, create_instance = \
            fetch_cifar100(args=args,root=data_path, test_size=test_size)
        
    elif dataset_name == 'TinyImageNet':
        _check_and_raise_error(args.split_type, 'pre', 'split scenario')
        raw_train, raw_test, create_instance = \
            fetch_tinyimagenet(args=args,root=data_path, test_size=test_size)
        
    elif dataset_name == 'CINIC10':
        _check_and_raise_error(args.split_type, 'pre', 'split scenario')
        raw_train, raw_test, create_instance = \
            fetch_cinic10(args=args,root=data_path, test_size=test_size)
    
    elif dataset_name == 'AGNews':
        args.max_length = dataset_args["max_length"]
        _check_and_raise_error(args.split_type, 'pre', 'split scenario')
        raw_train, raw_test, create_instance = \
            fetch_agnews(args,root=data_path, test_size=test_size)
    
    elif dataset_name == 'Sent140':
        args.max_length = dataset_args["max_length"]
        _check_and_raise_error(args.split_type, 'pre', 'split scenario')
        raw_train, raw_test, create_instance = \
            fetch_sent140(args, root=data_path, test_size=test_size)
    
    elif dataset_name == 'IMDB':
        args.max_length = dataset_args["max_length"]
        _check_and_raise_error(args.split_type, 'pre', 'split scenario')
        raw_train, raw_test, create_instance = \
              fetch_imdb(args, root=data_path, test_size=test_size)

    elif dataset_name == 'Adult':
        _check_and_raise_error(args.split_type, 'pre', 'split scenario', False)
        _check_and_raise_error(args.eval_type, 'local', 'evaluation type', False)
        args.num_clients = 16
        client_datasets = fetch_adult(args, root=data_path, test_size=test_size)
    
    elif dataset_name == 'Heart':
        _check_and_raise_error(args.split_type, 'pre', 'split scenario', False)
        _check_and_raise_error(args.eval_type, 'local', 'evaluation type', False)
        client_datasets = fetch_heart(args, root=data_path, test_size=test_size)
        args.in_features = 13
        args.num_classes = 2
        args.num_clients = 4
    
    elif dataset_name == 'Gleam':
        _check_and_raise_error(args.split_type, 'pre', 'split scenario', False)
        _check_and_raise_error(args.eval_type, 'local', 'evaluation type', False)
        args.num_clients = 38
        args.seq_len = dataset_args["seq_len"]
        client_datasets = fetch_gleam(args, root=data_path, test_size=test_size)
    
    elif dataset_name == 'UCIHAR':
        _check_and_raise_error(args.split_type, 'pre', 'split scenario', False)
        _check_and_raise_error(args.eval_type, 'local', 'evaluation type', False)
        args.num_clients = 30
        client_datasets = fetch_ucihar(args, root=data_path, test_size=test_size)
    
    elif dataset_name == 'Purchase100':
        _check_and_raise_error(args.split_type, 'pre', 'split scenario')
        raw_train, raw_test, create_instance = \
            fetch_purchase100(args, root=data_path, test_size=test_size)
    
    elif dataset_name == 'Texas100':
        _check_and_raise_error(args.split_type, 'pre', 'split scenario')
        raw_train, raw_test, create_instance = \
            fetch_texas100(args, root=data_path, test_size=test_size)
    
    elif dataset_name == 'Creditcard':
        _check_and_raise_error(args.split_type, 'pre', 'split scenario')  
        raw_train, raw_test, create_instance = \
            fetch_creditcard(args, root=data_path, test_size=test_size)
    
    else: 
        print(f'Dataset `{dataset_name.upper()}` is not supported yet... please check!')
        exit()
    
    
    # method to construct per-client dataset
    def _construct_dataset(raw_train, client_indices):
        """Construct per-client dataset."""
        if args.eval_type == "global":
            train_indices, test_indices = stratified_split(raw_train, client_indices, test_size=0)
        else:
            train_indices, test_indices = stratified_split(raw_train, client_indices, test_size)
        
        train_set = create_instance(train_indices, raw_train)
        test_set = create_instance(test_indices, raw_train)

        return (train_set, test_set)
            
    # get split indices if None
    if split_map is None and args.split_type != 'pre':
        print(f'[SIMULATE] DataSplitType-{args.split_type.upper()})')
        split_map = simulate_split(args, raw_train, num_classes)
    
    # construct client datasets if None
    if client_datasets is None and args.split_type != 'pre':
        client_datasets = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(args.num_clients, os.cpu_count() - 1)) as workhorse:
            for idx, sample_indices in TqdmToLogger(enumerate(split_map.values()), total=len(split_map)):
                client_datasets.append(workhorse.submit(_construct_dataset, raw_train, sample_indices).result())
        
        ## when if assigning pre-defined test split as a local holdout set (just divided by the total number of clients)
        
        # if raw_test!=None:  
        #     holdout_sets = torch.utils.data.random_split(raw_test, [int(len(raw_test) / args.num_clients)  for _ in range(args.num_clients)])
        #     augmented_datasets = []
        #     for idx, client_dataset in enumerate(client_datasets): 
        #         augmented_datasets.append((client_dataset[0], holdout_sets[idx]))
        #     client_datasets = augmented_datasets
    
    gc.collect()

    return client_datasets, raw_test, args
