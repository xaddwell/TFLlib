import numpy as np
#import torch
from typing import List

def Euclidean(gradient_rec: List, gradient_data: List):
    objective = np.zeros((1,))
    for rec, data in zip(gradient_rec, gradient_data):
        objective += np.sum(np.power((rec - data), 2))
    return 0.5 * objective

def CosSim(gradient_rec: List, gradient_data: List):
    scalar_product = np.zeros((1,))
    rec_norm = np.zeros((1,))
    data_norm = np.zeros((1,))

    for rec, data in zip(gradient_rec, gradient_data):
        scalar_product += np.sum((rec * data))
        rec_norm += np.sum(np.power(rec, 2))
        data_norm += np.sum(np.power(data, 2))

    objective = 1 - scalar_product / (np.sqrt(rec_norm) * np.sqrt(data_norm))
    return objective
  
ObjectFn_list = ['Euclidean', 'CosSim']