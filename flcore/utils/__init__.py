

'''
Description: 
Version: 1.0
Author: Jiahao Chen
Date: 2024-05-20 20:04:54
LastEditors: Jiahao Chen
LastEditTime: 2024-07-06 19:55:37
'''
import os
import random
import pynvml
import torch
from tqdm import tqdm
import numpy as np
from .logger import LoggerBase


os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

def get_best_gpu(device_id=100, exclude_ids=[0,1]): # return gpu(torch.device) with largest free memory.
    pynvml.nvmlInit()
    deviceCount = pynvml.nvmlDeviceGetCount()
    
    if device_id <= 0:
        print("Using CPU to training phrase")
        return torch.device("cpu")
    elif device_id <= deviceCount:
        return torch.device("cuda:%d"%(device_id-1))
    
    deviceMemory = []
    for i in range(deviceCount):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        deviceMemory.append(mem_info.free)
    for i in exclude_ids:
        deviceMemory[i] = 0
    deviceMemory = np.array(deviceMemory, dtype=np.int64)
    best_device_index = np.argmax(deviceMemory)
    return torch.device("cuda:%d"%(best_device_index))

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)#让显卡产生的随机数一致
    torch.cuda.manual_seed_all(seed)#多卡模式下，让所有显卡生成的随机数一致？这个待验证
    np.random.seed(seed)#numpy产生的随机数一致
    random.seed(seed)
    # CUDA中的一些运算，如对sparse的CUDA张量与dense的CUDA张量调用torch.bmm()，它通常使用不确定性算法。
    # 为了避免这种情况，就要将这个flag设置为True，让它使用确定的实现。
    torch.backends.cudnn.deterministic = True
    # 设置这个flag可以让内置的cuDNN的auto-tuner自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。
    # 但是由于噪声和不同的硬件条件，即使是同一台机器，benchmark都可能会选择不同的算法。为了消除这个随机性，设置为 False
    torch.backends.cudnn.benchmark = False


