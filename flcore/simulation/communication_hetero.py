'''
Description: 
Version: 1.0
Author: Jiahao Chen
Date: 2024-05-20 20:05:09
LastEditors: Jiahao Chen
LastEditTime: 2024-06-28 14:29:34
'''
# this means that for different client with different network conditions, 
# we can simulate the communication delay and bandwidth.

import random


def assign_network_condition(hetero_ratio=0.5):
    # simulate the network condition of different 
    network_condition = 1-random.normalvariate(0,1-hetero_ratio)
    # clamp the value of drop_rate between 0 and 1
    network_condition = max(0,min(1,network_condition))
    return network_condition


def simulate_communication_hetero(network_condition=1):
    # simulate the communication delay and bandwidth based on the network condition
    # we give the drop rate of a client for each epoch
    drop_rate = random.normalvariate(0,1-network_condition)
    # clamp the value of drop_rate between 0 and 1
    drop_rate = max(0,min(1,drop_rate))
    return drop_rate


def is_dropout(network_condition=1):
    # return True if the client is dropped out, otherwise return False
    drop_rate = simulate_communication_hetero(network_condition)
    return random.random() < drop_rate
    


if __name__=="__main__":

    network_condition = assign_network_condition(0.9)
    drop_rate = simulate_communication_hetero(network_condition,0.1)

    print(drop_rate)