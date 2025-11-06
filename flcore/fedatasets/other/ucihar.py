

import os
import torch
import logging
import torchtext
import pandas as pd
import numpy as np
from copy import deepcopy
from ..basedataset import TabularDataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)



class UCIHAR(TabularDataset):
    def __init__(self, dataset):
        super(UCIHAR,self).__init__(
            dataset[0], dataset[1]
        )


def create_instance(indices, original_dataset):
    selected_data = [deepcopy(original_dataset[i][0]) for i in indices]
    selected_targets = [deepcopy(original_dataset[i][1]) for i in indices]
    return UCIHAR((selected_data, selected_targets))


# helper method to fetch Heart disease classification dataset
def fetch_ucihar(args, root, test_size):

    str_folder = root
    INPUT_SIGNAL_TYPES = [
        "body_acc_x_",
        "body_acc_y_",
        "body_acc_z_",
        "body_gyro_x_",
        "body_gyro_y_",
        "body_gyro_z_",
        "total_acc_x_",
        "total_acc_y_",
        "total_acc_z_",
    ]

    str_train_files = [
        str_folder + "/train/" + "Inertial Signals/" + item + "train.txt"
        for item in INPUT_SIGNAL_TYPES
    ]
    str_test_files = [
        str_folder + "/test/" + "Inertial Signals/" + item + "test.txt"
        for item in INPUT_SIGNAL_TYPES
    ]
    str_train_y = str_folder + "/train/y_train.txt"
    str_test_y = str_folder + "/test/y_test.txt"
    str_train_id = str_folder + "/train/subject_train.txt"
    str_test_id = str_folder + "/test/subject_test.txt"

    def format_data_x(datafile):
        x_data = None
        for item in datafile:
            item_data = np.loadtxt(item, dtype=np.float32)
            if x_data is None:
                x_data = np.zeros((len(item_data), 1))
            x_data = np.hstack((x_data, item_data))
        x_data = x_data[:, 1:]
        X = None
        for i in range(len(x_data)):
            row = np.asarray(x_data[i, :])
            row = row.reshape(9, 128).T
            if X is None:
                X = np.zeros((len(x_data), 128, 9))
            X[i] = row
        return X

    def format_data_y(datafile):
        return np.loadtxt(datafile, dtype=np.int32) - 1

    def read_ids(datafile):
        return np.loadtxt(datafile, dtype=np.int32)

    X_train = format_data_x(str_train_files)
    X_test = format_data_x(str_test_files)
    Y_train = format_data_y(str_train_y)
    Y_test = format_data_y(str_test_y)
    id_train = read_ids(str_train_id)
    id_test = read_ids(str_test_id)

    X_train, X_test = X_train.reshape((-1, 9, 1, 128)), X_test.reshape((-1, 9, 1, 128))
    X = np.concatenate((X_train,X_test))
    Y = np.concatenate((Y_train,Y_test))

    scaler = StandardScaler()
    X = scaler.fit_transform(X)


    data_indices = np.concatenate((id_train,id_test))
    id_train_unique = np.unique(id_train)
    id_test_unique = np.unique(id_test)
    uids = np.unique(np.concatenate((id_train_unique, id_test_unique)))

    client_datasets = []
    for uid in uids:
        id_indices = np.where(data_indices == uid)[0]
        test_num = int(test_size*len(id_indices))
        train_indices = id_indices[test_num:]
        test_indices = id_indices[:test_num]
        trainset = UCIHAR((X[train_indices],Y[train_indices].flatten()))
        testset = UCIHAR((X[test_indices],Y[test_indices].flatten()))
        client_datasets.append((trainset,testset))
    
    return client_datasets