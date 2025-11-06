

import os
import sys
import time
import subprocess
from tqdm import tqdm
import yaml
import argparse



def get_base_cfg(datasets):
    base_cfg = {
        "CIFAR10": {
            "exp_name": "bkd",
            "task_id": "",
            "batch_size": 64,
            "local_epochs": 5,
            "model_name": "resnet18",
            "data_name": "CIFAR10",
            "num_clients": 100,
            "join_ratio": 0.1,
            "local_lr": 0.01,
            "optim": "sgd",
            "global_rounds": 2001,
            "save_interval": 200,
            "eval_type": "global",
            "split_type": "diri",
            "cncntrtn": 0.9,
            "dev_hetero": 0.9,
            "comm_hetero": 0.9,
            "bd_attack": "None",
            "bd_clients": 0,
            "bd_fixed": 0,
        },"FEMNIST": {
            "exp_name": "bkd",
            "task_id": "",
            "batch_size": 64,
            "local_epochs": 5,
            "model_name": "resnet18",
            "data_name": "FEMNIST",
            "num_clients": 100,
            "join_ratio": 0.1,
            "local_lr": 0.001,
            "optim": "sgd",
            "global_rounds": 2001,
            "save_interval": 200,
            "split_type": "pre",
            "eval_type": "local",
            "dev_hetero": 0.9,
            "comm_hetero": 0.9,
            "bd_attack": "None",
            "bd_clients": 0,
            "bd_fixed": 0,

        },"Texas100": {
            "exp_name": "bkd",
            "task_id": "",
            "batch_size": 64,
            "local_epochs": 5,
            "model_name": "logreg",
            "data_name": "Texas100",
            "num_clients": 100,
            "join_ratio": 0.1,
            "local_lr": 0.01,
            "optim": "sgd",
            "global_rounds": 501,
            "save_interval": 50,
            "eval_type": "global",
            "split_type": "diri",
            "cncntrtn": 0.9,
            "dev_hetero": 0.9,
            "comm_hetero": 0.9,
            "bd_attack": "None",
            "bd_clients": 0,
            "bd_fixed": 0,
        },"Purchase100": {
            "exp_name": "bkd",
            "task_id": "",
            "batch_size": 64,
            "local_epochs": 5,
            "model_name": "logreg",
            "data_name": "Purchase100",
            "num_clients": 100,
            "join_ratio": 0.1,
            "local_lr": 0.01,
            "optim": "sgd",
            "global_rounds": 501,
            "save_interval": 50,
            "eval_type": "global",
            "split_type": "diri",
            "cncntrtn": 0.9,
            "dev_hetero": 1,
            "comm_hetero": 1,
            "bd_attack": "None",
            "bd_clients": 0,
            "bd_fixed": 0,
        }, "IMDB": {
            "exp_name": "bkd",
            "task_id": "",
            "batch_size": 64,
            "local_epochs": 5,
            "model_name": "tinybert",
            "data_name": "IMDB",
            "num_clients": 100,
            "join_ratio": 0.1,
            "local_lr": 0.0001,
            "optim": "adamw",
            "global_rounds": 501,
            "save_interval": 50,
            "eval_type": "global",
            "split_type": "diri",
            "cncntrtn": 0.9,
            "dev_hetero": 0.9,
            "comm_hetero": 0.9,
            "bd_attack": "None",
            "bd_clients": 0,
            "bd_fixed": 0,
        },"AGNews": {
            "exp_name": "bkd",
            "task_id": "",
            "batch_size": 64,
            "local_epochs": 5,
            "model_name": "tinybert",
            "data_name": "AGNews",
            "num_clients": 100,
            "join_ratio": 0.1,
            "local_lr": 0.0001,
            "optim": "adamw",
            "global_rounds": 501,
            "save_interval": 50,
            "eval_type": "global",
            "split_type": "diri",
            "cncntrtn": 0.9,
            "dev_hetero": 0.9,
            "comm_hetero": 0.9,
            "bd_attack": "None",
            "bd_clients": 0,
            "bd_fixed": 0,
        }
    }

    return base_cfg[datasets]



def run_experiment(args):

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
            "--bd_attack", args["bd_attack"],
            "--bd_clients", str(args["bd_clients"]),
            "--bd_fixed", str(args["bd_fixed"]),
            "--exp_name", args["exp_name"],
            "--task_id", args["task_id"]
        ]
        
        command = [str(arg) for arg in command if arg]
        print('Running command:', command)
        result = subprocess.run(command, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("Script output:", result.stdout)
            break
        else:
            print(f"Error running script: {result.stderr}")




if __name__ == "__main__":

    time_str = time.strftime("%Y-%m-%d", time.localtime(time.time()))
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="AGNews")
    parser.add_argument("--split_type", type=str, default="diri")
    parser.add_argument("--cncntrtn", type=float, default=0.9)
    parser.add_argument("--comm_hetero", type=float, default=0.9)
    parser.add_argument("--dev_hetero", type=float, default=0.9)
    parser.add_argument("--bd_attack", type=str, default="None")
    parser.add_argument("--poison_ratio", type=int, default=-1)
    args = parser.parse_args()
    args.exp_name = f"{args.exp_name}_{time_str}"
    os.makedirs(f"./log/{args.exp_name}", exist_ok=True)

    bkd_attack_types = ['A3FL', 'DBA', 'Cerp', 'EdgeCase', 'Replace', 'Neurotoxin']
    datasets = ['AGNews', 'IMDB', 'CIFAR10', 'FEMNIST', 'Texas100', 'Purchase100']
    poison_ratios = [1, 3, 5, 7, 10]
    bkd_attack_types = [args.bd_attack] if args.bd_attack != "None" else bkd_attack_types
    poison_ratios = [args.poison_ratio] if args.poison_ratio != -1 else poison_ratios

    # for dataset in datasets:
    dataset = args.dataset
    base_cfg = get_base_cfg(dataset)
    base_cfg["exp_name"] = args.exp_name
    base_cfg["split_type"] = args.split_type
    base_cfg["cncntrtn"] = args.cncntrtn
    base_cfg["comm_hetero"] = args.comm_hetero
    base_cfg["dev_hetero"] = args.dev_hetero

    split_type = base_cfg["split_type"] if base_cfg["split_type"] in ["pre", "iid"] else f"diri{base_cfg['cncntrtn']}"
    comm_hetero = f"com{str(base_cfg['comm_hetero'])}"
    dev_hetero = f"dev{str(base_cfg['dev_hetero'])}"
    hetero_type = f"{split_type}_{comm_hetero}_{dev_hetero}"
    fixed_type = "fixed" if base_cfg['bd_fixed'] else "random"

    base_cfg["task_id"] = f"{dataset}_{hetero_type}_none"
    # run_experiment(base_cfg)

    for bd_attack in bkd_attack_types:
        base_cfg["bd_attack"] = bd_attack
        if bd_attack in ["A3FL", "Cerp"] and dataset not in ["CIFAR10", "FEMNIST"]:
            continue
        for poison_ratio in poison_ratios:
            base_cfg["bd_clients"] = poison_ratio
            task_id = f"{dataset}_{bd_attack}_{poison_ratio}_{fixed_type}_{hetero_type}"
            base_cfg["task_id"] = task_id
            run_experiment(base_cfg)
