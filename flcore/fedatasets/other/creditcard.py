import os
import torch
import logging
import torchtext
import pandas as pd
import numpy as np
from copy import deepcopy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from ..basedataset import TabularDataset

logger = logging.getLogger(__name__)




class Creditcard(TabularDataset):
    """
    Create a Creditcard dataset based on features and labels
    """

    def __init__(self, dataset):
        super(Creditcard, self).__init__(dataset[0], dataset[1])

    def __getitem__(self, index):
        data = self.data[index]
        targets = self.targets[index]
        data = torch.tensor(data).to(torch.float)
        targets = torch.tensor(targets).to(torch.long)
        return data, targets


def create_instance(indices, original_dataset):
    selected_data = [deepcopy(original_dataset[i][0]) for i in indices]
    selected_targets = [deepcopy(original_dataset[i][1]) for i in indices]
    return Creditcard((selected_data, selected_targets))



# helper method to fetch Adult dataset
def fetch_creditcard(args, root, test_size):
    file_path = os.path.join(root, "creditcard.csv")
    if not os.path.exists(file_path):
        raise ValueError("not such dataset")
    else:
        df = pd.read_csv(file_path).to_numpy()
    
    X, Y = df[:,1:29], df[:,-1]

    raw_train = Creditcard((X, Y))

    if args.eval_type == "global":
        indices = torch.randperm(len(raw_train))
        test_num = int(len(raw_train) * test_size)
        train_indices = indices[test_num:]
        test_indices = indices[:test_num]
        raw_test = create_instance(test_indices, raw_train)
        raw_train = create_instance(train_indices, raw_train)
    else:
        raw_test = Creditcard((None, None))
    
    return raw_train, raw_test, create_instance
