
import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dataset_root = r"xxx/TFLlib/data"
celeba_img_path = f"{dataset_root}/celeba/celeba/raw/img_align_celeba"

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--exp_name", type=str, default="Federated Exp")
    parser.add_argument("--task_id", type=str, default="")
    parser.add_argument("--seed", type=int, default=42, help='random seed')
    
    parser.add_argument("--model_name", type=str, default="resnet18", help="model used in task")
    parser.add_argument("--data_name", type=str, default="CIFAR10", help="dataset used in task")
    parser.add_argument("--optim", type=str, default="sgd", choices=['sgd', 'adamw'], help="optimizer used in task")
    parser.add_argument("--test_size", type=float, default=0.1, help='the ratio of data used to test model')
    parser.add_argument("--gpu", type=int, default=9, help='The total number of GPUs used in training. 0 means CPU.')

    parser.add_argument('-algo', "--algorithm", type=str, default="FedAvg")
    parser.add_argument('-nc', "--num_clients", type=int, default=100, help="Total number of clients")
    parser.add_argument('-jr', "--join_ratio", type=float, default=0.1, help="Ratio of clients per round")
    parser.add_argument('-rjr', "--random_join_ratio", type=bool, default=False, help="Random ratio of clients per round")
    
    parser.add_argument('-lbs', "--batch_size", type=int, default=64)
    parser.add_argument('-lr', "--local_lr", type=float, default=0.005, help="Local learning rate")
    parser.add_argument('-ld', "--lr_decay", action='store_true')
    parser.add_argument('-ldg', "--lr_decay_gamma", type=float, default=0.99)
    parser.add_argument('-gr', "--global_rounds", type=int, default=500)
    parser.add_argument('-ls', "--local_epochs", type=int, default=2, help="local epoch.")

    parser.add_argument("--eval_type", type=str, default="global", help="eval on local or global")

    parser.add_argument('-t', "--times", type=int, default=1, help="Running times")
    parser.add_argument('-eg', "--eval_gap", type=int, default=1, help="Rounds gap for evaluation")
    parser.add_argument('-sfn', "--save_folder_name", type=str, default='save')
    parser.add_argument('-nw', "--num_workers", type=int, default=1, help="The number of workers for data loading")
    

    ################## data, communication and system configuration #########################
    parser.add_argument("--split_type", type=str, default="iid", help='patho, diri, pre, unbalanced, iid')
    parser.add_argument("--mincls", type=int, default=2, help='the minimum number of distinct classes per client (valid only if `split_type` is `patho` or `diri`)')
    parser.add_argument("--cncntrtn", type=float, default=0.1, help='a concentration parameter for Dirichlet distribution (valid only if `split_type` is `diri`)')

    parser.add_argument('--comm_hetero', type=float, default=0.5, help="hetero of communication, from 0 to 1, 1 means off")
    parser.add_argument('--dev_hetero', type=float, default=0.5, help="hetero of device, from 0 to 1, 1 means off")
    #########################################################################################
    parser.add_argument("--save_interval", type=int, default=200, help="the interval of saving ckpt")
    parser.add_argument("--resume_ckpt", type=str, default=None, help='The path of global model checkpoint, which is the start point of training')
    parser.add_argument("--clip_norm", action='store_true', help='Wheather or not to clip norm during aggregation')
    parser.add_argument("--clip_factor", type=float, default=1, help="scale factor of clip norm")

    ################## attack configuration #########################
    parser.add_argument("--bd_start_round", type=int, default=0, help="the round to start backdoor attack")
    parser.add_argument("--bd_attack", type=str, default="DBA",  help='the name of the backdoor attack')
    parser.add_argument("--bd_clients", type=int, default=0,  help='the num of backdoor client')
    parser.add_argument("--bd_fixed", type=int, default=0,  help='whether keep attack 0 for random sample, 1 for fixed attack')

    parser.add_argument("--bd_lr", type=float, default=0.005, help="the learning rate of backdoor attacker")
    parser.add_argument("--bd_epoch", type=int, default=6, help="the local epoch of backdoor attacker")

    parser.add_argument("--bzt_attack", type=str, default="LIE",  help='the name of the byzantine attack')
    parser.add_argument("--bzt_clients", type=int, default=0,  help='the num of byzantine client')
    parser.add_argument("--bzt_fixed", type=int, default=0,  help='whether keep attack 0 for random sample, 1 for fixed attack')

    parser.add_argument("--privacy_attack", type=str, default=None, help="whether or not carry out privacy attack, you need to pass a string of attack name")
    parser.add_argument("--data_train", type=str, default='all', choices=['all', 'single_batch'], help='The way client trains its local data')
    parser.add_argument("--cid", type=int, default=0, help="Given the target cid of inference attack")
    parser.add_argument("--timeout", type=int, default=10, help="Time to wait during communication (s)")

    args = parser.parse_args()
    return args


    