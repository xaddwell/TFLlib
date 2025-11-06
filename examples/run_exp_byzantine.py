

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
            "split_type": "iid",
            "cncntrtn": 1,
            "dev_hetero": 1,
            "comm_hetero": 1,
            "bzt_attack": "None",
            "bzt_clients": 0,
            "bzt_fixed": 0,
        },"FEMNIST": {
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
            "dev_hetero": 1,
            "comm_hetero": 1,
            "bzt_attack": "None",
            "bzt_clients": 0,
            "bzt_fixed": 0,
        },"Texas100": {
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
            "split_type": "iid",
            "cncntrtn": 1,
            "dev_hetero": 1,
            "comm_hetero": 1,
            "bzt_attack": "None",
            "bzt_clients": 0,
            "bzt_fixed": 0,
        },"Purchase100": {
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
            "split_type": "iid",
            "cncntrtn": 1,
            "dev_hetero": 1,
            "comm_hetero": 1,
            "bzt_attack": "None",
            "bzt_clients": 0,
            "bzt_fixed": 0,
        }, "IMDB": {
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
            "split_type": "iid",
            "cncntrtn": 1,
            "dev_hetero": 1,
            "comm_hetero": 1,
            "bzt_attack": "None",
            "bzt_clients": 0,
            "bzt_fixed": 0,
        },"AGNews": {
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
            "split_type": "iid",
            "cncntrtn": 1,
            "dev_hetero": 1,
            "comm_hetero": 1,
            "bzt_attack": "None",
            "bzt_clients": 0,
            "bzt_fixed": 0,
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
            "--bzt_attack", args["bzt_attack"],
            "--bzt_clients", str(args["bzt_clients"]),
            "--bzt_fixed", str(args["bzt_fixed"]),
            "--exp_name", args["exp_name"],
            "--task_id", args["task_id"]
        ]
        
        command = [arg for arg in command if arg]
        print('Running command:', " ".join(command))
        result = subprocess.run(command, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("Script output:", result.stdout)
            break
        else:
            print(f"Error running script: {result.stderr}")


if __name__ == "__main__":


    time_str = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, required=True)
    args = parser.parse_args()
    args.exp_name = f"{args.exp_name}_{time_str}"
    os.makedirs(f"/home/zzm/RFLlib/log/{args.exp_name}", exist_ok=True)

    bzt_attack_types = ['Fang', 'IPM', 'LabelFlip', 'SignFlip', 'UpdateFlip', 'MinMax', 'LIE', 'MedianTailored', 'Noise', "SignGuard"]
    datasets = ['AGNews', 'IMDB', 'CIFAR10', 'Texas100', 'Purchase100']
    poison_ratios = [1, 2, 5, 10]
    
    for dataset in datasets:
        base_cfg = get_base_cfg(dataset)
        base_cfg["exp_name"] = args.exp_name

        split_type = base_cfg["split_type"] if base_cfg["split_type"] in ["pre", "iid"] else f"diri{base_cfg['cncntrtn']}"
        comm_hetero = f"com{str(base_cfg['comm_hetero'])}"
        dev_hetero = f"dev{str(base_cfg['dev_hetero'])}"
        hetero_type = f"{split_type}_{comm_hetero}_{dev_hetero}"
        fixed_type = "fixed" if base_cfg['bzt_fixed'] else "random"

        base_cfg["task_id"] = f"{dataset}_{hetero_type}_none"
        run_experiment(base_cfg)
        for bzt_attack in bzt_attack_types:
            base_cfg["bzt_attack"] = bzt_attack
            for poison_ratio in poison_ratios:

                base_cfg["bzt_clients"] = poison_ratio
                task_id = f"{dataset}_{bzt_attack}_{poison_ratio}_{fixed_type}_{hetero_type}"
                base_cfg["task_id"] = task_id
                run_experiment(base_cfg)
