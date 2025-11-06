'''
Description: 
Version: 1.0
Author: Jiahao Chen
Date: 2024-06-28 12:50:54
LastEditors: Jiahao Chen
LastEditTime: 2024-06-28 13:20:51
'''
import math
import random


def assign_device_condition(hetero_ratio):
    # assign device rate for each device
    device_rate = random.normalvariate(0, 1 - hetero_ratio)
    # clamp the value of device_rate between 0 and 1
    device_rate = max(0, min(1, 1 + device_rate))
    return device_rate



def device_hetero(device_rate, data_ratio, max_epochs):
    # simulate the device heterogeneity
    local_epoch = round(device_rate / data_ratio * max_epochs)
    return max(1, local_epoch)