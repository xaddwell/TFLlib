import torch
import random
import numpy as np
from tqdm import tqdm
import logging
from collections import defaultdict



logger = logging.getLogger(__name__)



def dirichlet_split_noniid(train_labels, alpha, n_clients):
    '''
    按照参数为alpha的Dirichlet分布将样本索引集合划分为n_clients个子集
    '''
    train_labels = np.array(train_labels)
    n_classes = max(train_labels)+1
    # (K, N) 类别标签分布矩阵X，记录每个类别划分到每个client去的比例
    label_distribution = np.random.dirichlet([alpha]*n_clients, n_classes)
    # (K, ...) 记录K个类别对应的样本索引集合
    class_idcs = [np.argwhere(train_labels == y).flatten()
                  for y in range(n_classes)]

    # 记录N个client分别对应的样本索引集合
    client_idcs = [[] for _ in range(n_clients)]
    for k_idcs, fracs in zip(class_idcs, label_distribution):
        # np.split按照比例fracs将类别为k的样本索引k_idcs划分为了N个子集
        # i表示第i个client，idcs表示其对应的样本索引集合idcs
        for i, idcs in enumerate(np.split(k_idcs,
                                          (np.cumsum(fracs)[:-1]*len(k_idcs)).
                                          astype(int))):
            client_idcs[i] += [idcs]

    client_idcs = [np.concatenate(idcs) for idcs in client_idcs]

    return client_idcs


class TqdmToLogger(tqdm):
    def __init__(self, *args, logger=None, 
    mininterval=0.1, 
    bar_format='{desc:<}{percentage:3.0f}% |{bar:20}| [{n_fmt:6s}/{total_fmt}]', 
    desc=None, 
    **kwargs
    ):
        self._logger = logger
        super().__init__(*args, mininterval=mininterval, bar_format=bar_format, desc=desc, **kwargs)

    @property
    def logger(self):
        if self._logger is not None:
            return self._logger
        return logger

    def display(self, msg=None, pos=None):
        if not self.n:
            return
        if not msg:
            msg = self.__str__()
        self.logger.info('%s', msg.strip('\r\n\t '))


def stratified_split(raw_dataset, sample_indices, test_size):
    """
    对给定的样本索引进行分层抽样划分训练和测试集。
    
    :param raw_dataset: 原始数据集对象
    :param sample_indices: 当前客户端的样本索引
    :param test_size: 测试集比例 (0 < test_size < 1)
    :return: 训练集索引列表，测试集索引列表
    """
    # 按标签聚合样本索引
    indices_per_label = defaultdict(list)
    for idx in sample_indices:
        label = int(raw_dataset.targets[idx])
        indices_per_label[label].append(idx)  # 使用全局索引
    
    # 分层抽样生成训练集和测试集索引
    train_indices, test_indices = [], []
    for label, indices in indices_per_label.items():
        n_test_samples = round(len(indices) * test_size)
        # 从每个类别中随机抽取测试样本
        test_samples = random.sample(indices, n_test_samples)
        test_indices.extend(test_samples)
        
        # 剩余样本作为训练集
        train_indices.extend(set(indices) - set(test_samples))

    return train_indices, test_indices


# def stratified_split(raw_dataset, sample_indices, test_size):
#     indices_per_label = defaultdict(list)
#     for index, label in enumerate(np.array(raw_dataset.targets)[sample_indices]):
#         indices_per_label[label.item()].append(index)
    
#     train_indices, test_indices = [], []
#     for label, indices in indices_per_label.items():
#         n_samples_for_label = round(len(indices) * test_size)
#         random_indices_sample = random.sample(indices, n_samples_for_label)
#         test_indices.extend(random_indices_sample)
#         train_indices.extend(set(indices) - set(random_indices_sample))
#     return train_indices, test_indices
