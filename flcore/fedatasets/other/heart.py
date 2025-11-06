import os
import torch
import logging
import torchtext
import pandas as pd
from copy import deepcopy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

from ..basedataset import TabularDataset

class Heart(TabularDataset):
    """
    Create a Heart dataset based on features and labels
    """

    def __init__(self, dataset):
        super(Heart, self).__init__(dataset[0], dataset[1])

    def __getitem__(self, index):
        data = self.data[index]
        targets = self.targets[index]
        data = torch.tensor(data).to(torch.float)
        targets = torch.tensor(targets).to(torch.long)
        return data, targets


def create_instance(indices, original_dataset):
    selected_data = [deepcopy(original_dataset[i][0]) for i in indices]
    selected_targets = [deepcopy(original_dataset[i][1]) for i in indices]
    return Heart((selected_data, selected_targets))


# helper method to fetch Heart disease classification dataset
def fetch_heart(args, root, test_size):
    URL = {
        'cleveland': 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data',
        'hungarian': 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.hungarian.data',
        'switzerland': 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.switzerland.data',
        'va': 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.va.data'
    } 
    MD5 = {
        'cleveland': '2d91a8ff69cfd9616aa47b59d6f843db',
        'hungarian': '22e96bee155b5973568101c93b3705f6',
        'switzerland': '9a87f7577310b3917730d06ba9349e20',
        'va': '4249d03ca7711e84f4444768c9426170'
    } 
    COL_NAME = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'targets']
    
    def _download(root):
        for hospital in URL.keys():
            _ = torchtext.utils.download_from_url(
                url=URL[hospital], 
                root=root, 
                hash_value=MD5[hospital], 
                hash_type='md5'
            )
            os.rename(os.path.join(root, URL[hospital].split('/')[-1]), os.path.join(root, f'HEART ({hospital}).csv'))
    
    def _munge_and_split(root, hospital, test_size):
        # load data
        to_drop = [
            10, # the slope of the peak exercise ST segment
            11, # number of major vessels (0-3) colored by flourosopy
            12 # thalassemia background
        ]
        df = pd.read_csv(os.path.join(root, f'HEART ({hospital}).csv'), header=None, na_values='?', usecols=[i for i in range(14) if i not in to_drop]).apply(lambda x: x.fillna(x.mean()),axis=0).reset_index(drop=True)
        
        # rename column
        df.columns = COL_NAME
        
        # adjust dtypes
        df['targets'] = df['targets'].where(df['targets'] == 0, 1)
        df['age'] = df['age'].astype(float)
        df['sex'] = df['sex'].astype(int)
        df['cp'] = df['cp'].astype(int)
        df['trestbps'] = df['trestbps'].astype(float)
        df['chol'] = df['chol'].astype(float)
        df['restecg'] = df['restecg'].astype(int)
        df['cp'] = df['cp'].astype(int)
        df['thalach'] = df['thalach'].astype(float)
        df['exang'] = df['exang'].astype(int)
        df['oldpeak'] = df['oldpeak'].astype(float)
        
        # get one-hot encoded dummy columns for categorical data
        df = pd.concat([pd.get_dummies(df.iloc[:, :-1], columns=['cp', 'restecg'], drop_first=True, dtype=int), df[['targets']]], axis=1)
        
        # get inputs and targets
        inputs, targets = df.iloc[:, :-1].values, df.iloc[:, -1].values
        
        # train-test split with stratified manner
        train_inputs, test_inputs, train_targets, test_targets = train_test_split(inputs, targets, test_size=test_size, random_state=args.seed, stratify=targets)
        
        # scaling inputs
        scaler = StandardScaler()
        train_inputs = scaler.fit_transform(train_inputs)
        test_inputs = scaler.transform(test_inputs)
        return (Heart((train_inputs, train_targets)), Heart((test_inputs, test_targets))) 
        
    if not os.path.exists(os.path.join(root, 'heart')):
        _download(root=os.path.join(root, 'heart'))
    else:
        logger.info(f'[LOAD] [HEART] ...raw data already exists!')
    
    client_datasets = []
    for hospital in URL.keys():
        client_datasets.append(_munge_and_split(os.path.join(root, 'heart'), hospital, test_size))
    
    
    return client_datasets
