import os
import torch
import logging
import torchvision
from ..utils.transform import transform_torch as dataset_transform
from ..basedataset import VisionDataset

logger = logging.getLogger(__name__)


class CIFAR100(VisionDataset):

    def __init__(self, dataset):
        super(CIFAR100, self).__init__(dataset[0], dataset[1])

    def __getitem__(self, index):
        data = self.data[index]
        targets = self.targets[index]
        data = torch.tensor(data).to(torch.float)
        targets = torch.tensor(targets).to(torch.long)
        return data, targets


def create_instance(indices, original_dataset):
    selected_data = [original_dataset[i][0].clone() for i in indices]
    selected_targets = [original_dataset[i][1].clone() for i in indices]
    return CIFAR100((selected_data, selected_targets))

    
        
# helper method to fetch CINIC-10 dataset
def fetch_cifar100(args, root, test_size):
    
    # load CINIC-10 with ImageFolder
    DEFAULT_ARGS = {'root': root, 'train': True, 'download': True,
                    'transform': dataset_transform["CIFAR100"]["train"]}
    raw_train = torchvision.datasets.CIFAR100(**DEFAULT_ARGS)
    DEFAULT_ARGS['train'] = False
    DEFAULT_ARGS['transform'] = dataset_transform["CIFAR100"]["test"]
    raw_test = torchvision.datasets.CIFAR100(**DEFAULT_ARGS)


    trainloader = torch.utils.data.DataLoader(
        raw_train, batch_size=len(raw_train.data), shuffle=False)
    testloader = torch.utils.data.DataLoader(
        raw_test, batch_size=len(raw_test.data), shuffle=False)
    
    for _, train_data in enumerate(trainloader, 0):
        raw_train.data, raw_train.targets = train_data
    for _, test_data in enumerate(testloader, 0):
        raw_test.data, raw_test.targets = test_data

    # print(raw_test.data, raw_test.targets)

    # split set into data and labels

    # merge train and test set
    raw_train.data = torch.cat((raw_train.data, raw_test.data), 0)
    raw_train.targets = torch.cat((raw_train.targets, raw_test.targets), 0)

    raw_train = CIFAR100((raw_train.data, raw_train.targets))

    if args.eval_type == "global":
        indices = torch.randperm(len(raw_train))
        test_num = int(len(raw_train) * test_size)
        train_indices = indices[test_num:]
        test_indices = indices[:test_num]
        raw_test = create_instance(test_indices, raw_train)
        raw_train = create_instance(train_indices, raw_train)
    else:
        raw_test = CIFAR100((None, None))

    
    return raw_train, raw_test, create_instance
