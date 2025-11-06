'''
Description: 
Version: 1.0
Author: Jiahao Chen
Date: 2024-06-29 11:11:44
LastEditors: Jiahao Chen
LastEditTime: 2024-06-29 18:04:05
'''


import os
import torch
from flcore.config import dataset_root,celeba_img_path



dataset_dict = {
    "CIFAR10":{"num_classes":10, "in_channels":3, "domain":"cv"},
    
    "CINIC10":{"num_classes":10, "in_channels":3, "domain":"cv"},
    
    "CIFAR100":{"num_classes":100, "in_channels":3, "domain":"cv"},
    
    "FEMNIST":{"num_classes":62, "in_channels":1, "domain":"cv"},
    
    "CelebA":{"num_classes":2, "in_channels":3, "img_path":celeba_img_path, "domain":"cv"},
    
    "Sent140":{'num_embeddings': 400001,'seq_len': 25,"num_classes":2,"num_layers":2,'max_length': 64, "use_tokenizer":True,"embedding_size":300,"hidden_size":256,"domain":"nlp"},
    
    "AGNews":{"num_classes":4,"embedding_size":200,"dropout": 0.2,"num_embeddings":40000,"max_length":64,"use_tokenizer":True,"num_layers":2,"hidden_size":200,"is_seq2seq":False,"domain":"nlp"},
    
    "IMDB":{"num_classes":2,"embedding_size":200,"hidden_size":200,"num_layers":2,"dropout": 0.2,"max_length":64,"num_embeddings":40000,"use_tokenizer":True,"is_seq2seq":False,"domain":"nlp"},
  
    
    
    "Purchase100":{"num_classes":100,"in_features":600,"hidden_size":64,"domain":"tabular"},

    "Texas100":{"num_classes":100,"in_features":6169,"hidden_size":64,"domain":"tabular"},
    
    "Creditcard":{"num_classes":2,"in_features":28,"hidden_size":64,"domain":"tabular"},
    
    "UCIHAR":{"num_classes":6,"in_features":561,"hidden_size":64,"domain":"tabular"},
    
    "Adult":{"num_classes":2,"in_features":84,"hidden_size":64,"domain":"tabular"},
    
    "Gleam":{"num_classes":8,"in_features":14,"hidden_size":64,'seq_len':8, "domain":"tabular"}
}

