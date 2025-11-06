import os
import json
import torch
import logging
import random
import torchvision
from ..utils.transform import transform_torch as dataset_transform
from ..basedataset import VisionDataset, ImageDataset

logger = logging.getLogger(__name__)


class FEMNIST(VisionDataset):

    def __init__(self, dataset):
        super(FEMNIST, self).__init__(dataset[0], dataset[1])

    def __getitem__(self, index):
        data = self.data[index]
        targets = self.targets[index]
        data = torch.tensor(data).to(torch.float)
        targets = torch.tensor(targets).to(torch.long)
        return data, targets



def create_instance(indices, original_dataset):
    selected_data = [original_dataset[i][0].clone() for i in indices]
    selected_targets = [original_dataset[i][1].clone() for i in indices]
    return FEMNIST((selected_data, selected_targets))

    
        

def fetch_femnist(args, root, test_size):
    
    client_datasets = []
    transform = dataset_transform["FEMNIST"]["train"]
    client_list = os.listdir(root)
    json_file = os.path.join(root, "user_num.json")
    with open(json_file, "r") as f:
        data = json.load(f)

    user_num = dict(data)
    
    # 1. sort user_num by value
    # user_num = sorted(user_num.items(), key=lambda x: x[1], reverse=True)
    
    # 2. shuffle user_num
    user_num = random.sample(user_num.items(), len(user_num))
    

    for client_id in range(args.num_clients):
        client = user_num[client_id][0]
        # print("client: ", client, " has ", user_num[client_id][1], " samples")
        client_path = os.path.join(root, client)
        full_set = ImageDataset(client_path,transform=transform, mode="L")
        full_loader = torch.utils.data.DataLoader(full_set, batch_size=len(full_set), shuffle=False)
        for _, train_data in enumerate(full_loader, 0):
            data, targets = train_data
        
        full_set = FEMNIST((data, targets))
        id_indices = list(range(len(targets)))
        random.shuffle(id_indices)
        test_num = int(test_size*len(targets))
        trainset = create_instance(id_indices[test_num:], full_set)
        testset = create_instance(id_indices[:test_num], full_set)
        client_datasets.append((trainset, testset))

    return client_datasets
