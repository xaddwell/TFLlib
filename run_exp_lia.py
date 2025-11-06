

import os
import sys
import time
import subprocess
from tqdm import tqdm
import yaml
import argparse
import itertools    

def get_base_cfg(datasets):

    base_cfg = {
        "CIFAR10": {
            "model_name": "resnet18",
            "data_name": "CIFAR10",
            "join_ratio": 0.1,
            "local_lr": 0.01,
            "optim": "sgd",
            "global_rounds": 2001,
            "save_interval": 200,
            "eval_type": "global",
            "split_type": "iid",
            "cncntrtn": 1,
            "dev_hetero": 1,
            "comm_hetero": 1,
        },
        "FEMNIST": {
            "model_name": "resnet18",
            "data_name": "FEMNIST",
            "join_ratio": 0.1,
            "local_lr": 0.001,
            "optim": "sgd",
            "global_rounds": 2001,
            "save_interval": 200,
            "split_type": "pre",
            "eval_type": "local",
            "cncntrtn": 1,                                  
            "dev_hetero": 1,
            "comm_hetero": 1,
        },"Texas100": {
            "model_name": "logreg",
            "data_name": "Texas100",
            "join_ratio": 0.1,
            "local_lr": 0.01,
            "optim": "sgd",
            "global_rounds": 501,
            "save_interval": 50,
            "eval_type": "global",
            "split_type": "iid",
            "cncntrtn": 1,
            "dev_hetero": 1,
            "comm_hetero": 1
        },"Purchase100": {
            "model_name": "logreg",
            "data_name": "Purchase100",
            "join_ratio": 0.1,
            "local_lr": 0.01,
            "optim": "sgd",
            "global_rounds": 501,
            "save_interval": 50,
            "eval_type": "global",
            "split_type": "iid",
            "cncntrtn": 1,
            "dev_hetero": 1,
            "comm_hetero": 1,
        }, "IMDB": {
            "model_name": "tinybert",
            "data_name": "IMDB",
            "join_ratio": 0.1,
            "local_lr": 0.0001,
            "optim": "adamw",
            "global_rounds": 501,
            "save_interval": 50,
            "eval_type": "global",
            "split_type": "iid",
            "cncntrtn": 1,
            "dev_hetero": 1,
            "comm_hetero": 1,
        },"AGNews": {
            "model_name": "tinybert",
            "data_name": "AGNews",
            "join_ratio": 0.1,
            "local_lr": 0.0001,
            "optim": "adamw",
            "global_rounds": 501,
            "save_interval": 50,
            "eval_type": "global",
            "split_type": "iid",
            "cncntrtn": 1,
            "dev_hetero": 1,
            "comm_hetero": 1,
        }
    }

    return base_cfg[datasets]



def run_experiment(args):
    wait_time = 2
    while True:
        command = [
            "python", "main.py", 
            "--model_name", args["model_name"],
            "--data_name", args["data_name"],
            "--num_clients", str(args["num_clients"]),
            "--join_ratio", str(args["join_ratio"]),
            "--optim", args["optim"],
            "--batch_size", str(args["batch_size"]),
            "--local_lr", str(args["local_lr"]),
            "--local_epochs", str(args["local_epochs"]),
            "--global_rounds", str(args["global_rounds"]),
            "--eval_type", args["eval_type"],
            "--split_type", args["split_type"],
            "--cncntrtn", str(args["cncntrtn"]),
            "--dev_hetero", str(args["dev_hetero"]),
            "--comm_hetero", str(args["comm_hetero"]),
            "--save_interval", str(args["save_interval"]),
            "--lr_decay",
            "--exp_name", args["exp_name"],
            "--task_id", args["task_id"],
            "--data_train", "single_batch",
            "--privacy_attack", args["lia_attack_types"],
            "--cid", "0",
            "--global_rounds", str(args['global_rounds']),
        ]
        
        command = [arg for arg in command if arg]
        print('Running command:', " ".join(command))
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode == 0:
            print("Script output:", result.stdout)
            wait_time = 2
            break
        else:
            print(f"Error running script: {result.stderr}")
            wait_time -= 1
            if wait_time == 0:
                print("Max wait time reached. Exiting.")
                break


if __name__ == "__main__":
    time_str = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, required=True)
    args = parser.parse_args()
    args.exp_name = f"{args.exp_name}" #_{time_str
    os.makedirs(f"/home/zzm/RFLlib/log/{args.exp_name}", exist_ok=True)

    batch_sizes = [8, 16, 32, 64]
    local_epochs = [1, 2, 3, 4, 5]
    num_clients = [10, 100]
    global_rounds = [2]
    lia_attack_types = ['GI', 'RLG', 'iLRG']
    datasets = ['AGNews', 'IMDB', 'CIFAR10', 'Texas100', 'Purchase100']
    combinations = list(itertools.product(batch_sizes, local_epochs, num_clients, lia_attack_types, datasets, global_rounds))
    
    for batch_size, local_epoch, num_client, lia_attack_type, dataset, global_round in combinations:
        base_cfg = get_base_cfg(dataset)
        base_cfg["exp_name"] = args.exp_name
        base_cfg["lia_attack_types"] = lia_attack_type
        base_cfg["batch_size"] = batch_size
        base_cfg["local_epochs"] = local_epoch
        base_cfg["num_clients"] = num_client
        base_cfg["global_rounds"] = global_round

        split_type = base_cfg["split_type"] if base_cfg["split_type"] in ["pre", "iid"] else f"diri{base_cfg['cncntrtn']}"
        comm_hetero = f"com{str(base_cfg['comm_hetero'])}"
        dev_hetero = f"dev{str(base_cfg['dev_hetero'])}"
        hetero_type = f"{split_type}_{comm_hetero}_{dev_hetero}"

        task_id = f"{dataset}_{lia_attack_type}_bz{batch_size}_local_epoch-{local_epoch}_num_clients-{num_client}_gloval_round-{global_round}_{hetero_type}"
        if os.path.exists(f"/home/zzm/RFLlib/log/{args.exp_name}/{task_id}"):
            continue
        base_cfg["task_id"] = task_id
        run_experiment(base_cfg)
