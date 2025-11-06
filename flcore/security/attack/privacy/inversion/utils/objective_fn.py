from typing import List

def Euclidean(gradient_rec: List, gradient_data: List):
    objective = gradient_rec[0].new_zeros(1,)
    for rec, data in zip(gradient_rec, gradient_data):
        objective += (rec - data).pow(2).sum()
    return 0.5 * objective

def CosSim(gradient_rec: List, gradient_data: List):
    scalar_product = gradient_rec[0].new_zeros(1,)
    rec_norm = gradient_rec[0].new_zeros(1,)
    data_norm = gradient_rec[0].new_zeros(1,)

    for rec, data in zip(gradient_rec, gradient_data):
        scalar_product += (rec * data).sum()
        rec_norm += rec.pow(2).sum()
        data_norm += data.pow(2).sum()

    objective = 1 - scalar_product / (rec_norm.sqrt() * data_norm.sqrt())
    return objective
  
ObjectFn_list = ['Euclidean', 'CosSim']