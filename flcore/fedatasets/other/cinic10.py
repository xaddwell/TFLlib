import os
import torch
import logging
import torchvision
from ..utils.transform import transform_torch as dataset_transform
from ..basedataset import VisionDataset

logger = logging.getLogger(__name__)


class CINIC10(VisionDataset):
    
    base_folder = 'cinic10'
    zip_md5 ='6ee4d0c996905fe93221de577967a372'
    splits = ('train', 'val', 'test')
    filename = 'CINIC-10.tar.gz'
    url = 'https://datashare.is.ed.ac.uk/bitstream/handle/10283/3192/CINIC-10.tar.gz'

    def __init__(self, dataset):
        super(CINIC10, self).__init__(dataset[0], dataset[1])

    def __getitem__(self, index):
        data = self.data[index]
        targets = self.targets[index]
        data = torch.tensor(data).to(torch.float)
        targets = torch.tensor(targets).to(torch.long)
        return data, targets
    
    def download(self):
        if self._check_exists(): 
            return
        torchvision.datasets.utils.download_and_extract_archive(
            self.url, self.dataset_folder, filename=self.filename,
            remove_finished=True, md5=self.zip_md5
        )


def create_instance(indices, original_dataset):
    selected_data = [original_dataset[i][0].clone() for i in indices]
    selected_targets = [original_dataset[i][1].clone() for i in indices]
    return CINIC10((selected_data, selected_targets))

    
        
# helper method to fetch CINIC-10 dataset
def fetch_cinic10(args, root, test_size):
    
    # load CINIC-10 with ImageFolder
    DEFAULT_ARGS = {'root': os.path.join(root, 'train'),
                    'transform': dataset_transform["CINIC10"]["train"]}
    raw_train = torchvision.datasets.ImageFolder(**DEFAULT_ARGS)
    DEFAULT_ARGS['root'] = os.path.join(root, 'test')
    DEFAULT_ARGS['transform'] = dataset_transform["CINIC10"]["test"]
    raw_test = torchvision.datasets.ImageFolder(**DEFAULT_ARGS)

    trainloader = torch.utils.data.DataLoader(
        raw_train, batch_size=len(raw_train.data), shuffle=False)
    testloader = torch.utils.data.DataLoader(
        raw_test, batch_size=len(raw_test.data), shuffle=False)
    
    for _, train_data in enumerate(trainloader, 0):
        raw_train.data, raw_train.targets = train_data
    for _, test_data in enumerate(testloader, 0):
        raw_test.data, raw_test.targets = test_data
    # split set into data and labels

    # merge train and test set
    raw_train.data = torch.cat((raw_train.data, raw_test.data), 0)
    raw_train.targets = torch.cat((raw_train.targets, raw_test.targets), 0)

    raw_train = CINIC10((raw_train.data, raw_train.targets))

    if args.eval_type == "global":
        indices = torch.randperm(len(raw_train))
        test_num = int(len(raw_train) * test_size)
        train_indices = indices[test_num:]
        test_indices = indices[:test_num]
        raw_test = create_instance(test_indices, raw_train)
        raw_train = create_instance(train_indices, raw_train)
    else:
        raw_test = CINIC10((None, None))

    
    return raw_train, raw_test, create_instance
