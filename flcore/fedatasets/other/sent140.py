
import os
import torch
import spacy
import numpy as np
import pandas as pd
from datasets import load_dataset
from ..basedataset import TextDataset


class Sent140(TextDataset):
    def __init__(self, dataset, max_length):
        super(Sent140,self).__init__(
            dataset[0], dataset[1], max_length
        )



def create_instance(indices, original_dataset):
    selected_data = [original_dataset[i][0] for i in indices]
    selected_targets = [original_dataset[i][1] for i in indices]
    max_length = original_dataset.max_length if hasattr(original_dataset, 'max_length') else None
    return Sent140((selected_data, selected_targets), max_length)



# helper method to fetch Twitter dataset
def fetch_sent140(args, root, test_size):

    def load_data(split="train"):
        datasets = load_dataset("MrbBakh/Sentiment140",split=split)
        data = datasets["text"]
        targets = datasets["sentiment"]
        return data, targets

    train_data = load_data(split="train")
    test_data = load_data(split="test")

    # Merge train and test data
    all_data = train_data[0] + test_data[0]
    all_targets = train_data[1] + test_data[1]
    raw_train = Sent140((all_data, all_targets), args.max_length)

    if args.eval_type == "global":
        indices = torch.randperm(len(raw_train))
        test_num = int(len(raw_train) * test_size)
        train_indices = indices[test_num:]
        test_indices = indices[:test_num]
        raw_test = create_instance(test_indices, raw_train)
        raw_train = create_instance(train_indices, raw_train)
    else:
        raw_test = Sent140((None, None), args.max_length)
    
    return raw_train, raw_test, create_instance