


import os
import torch
import logging
import torchtext
import pandas as pd
import numpy as np
from copy import deepcopy
from ..basedataset import TabularDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)



class Purchase100(TabularDataset):
    """
    Create a Purchase100 dataset based on features and labels
    """

    def __init__(self, dataset):
        super(Purchase100, self).__init__(dataset[0], dataset[1])

    def __getitem__(self, index):
        data = self.data[index]
        targets = self.targets[index]
        data = torch.tensor(data).to(torch.float)
        targets = torch.tensor(targets).to(torch.long)
        return data, targets


def create_instance(indices, original_dataset):
    selected_data = [deepcopy(original_dataset[i][0]) for i in indices]
    selected_targets = [deepcopy(original_dataset[i][1]) for i in indices]
    return Purchase100((selected_data, selected_targets))



# helper method to fetch Adult dataset
def fetch_purchase100(args, root, test_size):

    file_path = os.path.join(root, "purchase100.npz")
    if not os.path.exists(file_path):
        dataset_path = os.path.join(root, "purchase100")
        data_set = np.genfromtxt(dataset_path, delimiter=",")
        logging.info("Finish processing the dataset.")
        X = data_set[:, 1:].astype(np.float64)
        Y = (data_set[:, 0]).astype(np.int32) - 1
        np.savez(os.path.join(root, "purchase100.npz"), X=X, Y=Y)
    else:
        dataset = np.load(file_path)
        ## randomly shuffle the data
        X, Y = dataset["X"], dataset["Y"]
    
    raw_train = Purchase100((X, Y))

    if args.eval_type == "global":
        indices = torch.randperm(len(raw_train))
        test_num = int(len(raw_train) * test_size)
        train_indices = indices[test_num:]
        test_indices = indices[:test_num]
        raw_test = create_instance(test_indices, raw_train)
        raw_train = create_instance(train_indices, raw_train)
    else:
        raw_test = Purchase100((None, None))    
    return raw_train, raw_test, create_instance