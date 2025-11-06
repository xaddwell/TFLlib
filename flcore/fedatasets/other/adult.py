import os
import torch
import logging
import torchtext
import pandas as pd

from ..basedataset import TabularDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)



class Adult(TabularDataset):
    def __init__(self, dataset):
        super(Adult, self).__init__(dataset[0], dataset[1])

    def __getitem__(self, index):
        data = torch.tensor(self.data[index]).float()
        targets = torch.tensor(self.targets[index]).long()
        return data, targets

# helper method to fetch Adult dataset
def fetch_adult(args, root, test_size):
    URL = [
        'http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data',
        'http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test'
    ]
    MD5 = [
        '5d7c39d7b8804f071cdd1f2a7c460872',
        '35238206dfdf7f1fe215bbb874adecdc'
    ]
    COL_NAME = [
        'age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status',\
        'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss',\
        'house_per_week', 'native_country', 'targets'
    ]
    NUM_COL = ['age', 'fnlwgt', 'capital_gain', 'capital_loss', 'house_per_week', 'education_num']
    
    def _download(root):
        for idx, (url, md5) in enumerate(zip(URL, MD5)):
            _ = torchtext.utils.download_from_url(
                url=url, 
                root=root, 
                hash_value=md5, 
                hash_type='md5'
            )
            os.rename(os.path.join(root, url.split('/')[-1]), os.path.join(root, f"adult_{'train' if idx == 0 else 'test'}.csv"))
    
    
        
    logger.info(f'[LOAD] [ADULT] Check if raw data exists; if not, start downloading!')
    if not os.path.exists(root):
        _download(root)
        logger.info(f'[LOAD] [ADULT] ...raw data is successfully downloaded!')
    else:
        logger.info(f'[LOAD] [ADULT] ...raw data already exists!')
    

    def _munge_and_create_clients(root):
        # load dat
        df = pd.read_csv(os.path.join(root, 'adult_train.csv'), header=None, names=COL_NAME, na_values='?').dropna().reset_index(drop=True)
        df = df.drop(columns=['education'])
    
        # encode categorical data
        for col in df.columns:
            if col not in NUM_COL:
                replace_map = {key: value for value, key in enumerate(sorted(df[col].unique()))}
                df[col] = df[col].replace(replace_map)

        # adjust dtype
        for col in df.columns:
            if col in NUM_COL:
                df[col] = df[col].astype('float')
            else:
                df[col] = df[col].astype('category')
        
        # get one-hot encoded dummy columns for categorical data
        df = pd.concat([pd.get_dummies(df.iloc[:, :-1], columns=[col for col in df.columns if col not in NUM_COL][:-1], drop_first=True, dtype=int), df[['targets']]], axis=1)
        
        # creat clients by education
        clients = {}
        for edu in df['education_num'].unique():
            clients[edu] = df.loc[df['education_num'] == edu]
        return clients
    
   
    raw_clients = _munge_and_create_clients(root)
    

    def _process_client_datasets(dataset, test_size):
        # remove identifier column
        edu = int(dataset['education_num'].unique()[0])
        df = dataset.drop(columns=['education_num'])
        inputs, targets = df.iloc[:, :-1].values, df.iloc[:, -1].values
        
        # train-test split with stratified manner
        train_inputs, test_inputs, train_targets, test_targets = train_test_split(inputs, targets, test_size=test_size, random_state=args.seed, stratify=targets)
        
        # scaling inputs
        scaler = MinMaxScaler()
        train_inputs[:, :5] = scaler.fit_transform(train_inputs[:, :5])
        test_inputs[:, :5] = scaler.transform(test_inputs[:, :5])
        return (Adult((train_inputs, train_targets)), Adult((test_inputs, test_targets))) 
    
    client_datasets = []
    for dataset in raw_clients.values():
        client_datasets.append(_process_client_datasets(dataset, test_size))

    return client_datasets
