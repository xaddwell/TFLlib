'''
Description: 
Version: 1.0
Author: Jiahao Chen
Date: 2024-06-29 22:14:25
LastEditors: Jiahao Chen
LastEditTime: 2024-06-29 22:14:35
'''


import torch.nn as nn

# split an original model into a base and a head
class BaseHeadSplit(nn.Module):
    def __init__(self, base, head):
        super(BaseHeadSplit, self).__init__()

        self.base = base
        self.head = head
        
    def forward(self, x):
        out = self.base(x)
        out = self.head(out)

        return out