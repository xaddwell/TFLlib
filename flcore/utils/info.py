import os
import math
import time
import yaml

def write_info(args):
    # 将 args 转换为字典
    args_dict = vars(args)

    # 将 args 字典保存为 YAML 文件
    with open(os.path.join(args.save_folder_name, 'args.yaml'), 'w') as file:
        yaml.dump(args_dict, file, default_flow_style=False, allow_unicode=True)
    

