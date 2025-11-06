
import os

import torch
from PIL import Image

class BaseDataset(torch.utils.data.Dataset):
    def __init__(self,**kwargs):
        super(BaseDataset, self).__init__()

    
    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        input, label = self.data[idx], self.targets[idx]

        return input, torch.LongTensor(label)



class VisionDataset(BaseDataset):
    def __init__(self, data, targets, transform=None):
        super(VisionDataset, self).__init__()
        self.data = data
        self.targets = targets
        self.transform = transform

    def __getitem__(self, idx):
        input, label = self.data[idx], self.targets[idx]
        
        if self.transform:
            input = self.transform(input)
        
        return input, torch.LongTensor(label)

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return f'VisionDataset(n_samples={len(self.data)})'



class TabularDataset(BaseDataset):
    def __init__(self, data, targets, transform=None):
        super(TabularDataset, self).__init__()
        self.data = data
        self.targets = targets
        self.transform = transform if transform else lambda x: x

    def __getitem__(self, idx):
        input, label = self.data[idx], self.targets[idx]
        if self.transform:
            input = self.transform(input)
        return torch.Tensor(input), label

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return f'TabularDataset(n_samples={len(self.data)})'



class TextDataset(BaseDataset):
    def __init__(self, data, targets, max_length):
        super(TextDataset, self).__init__()
        self.data = data
        self.targets = targets
        self.max_length = max_length
        self.tokenizer = None

    def __getitem__(self, idx):
        text, label = self.data[idx], self.targets[idx]
        
        if self.tokenizer is None:
             return text, label
        
        # Tokenize the text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Get the input_ids and attention_mask
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        label = torch.LongTensor([label])

        # print(input_ids.shape, attention_mask.shape, label.shape)

        return input_ids, attention_mask, label

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return f'TextDataset(n_samples={len(self.data)})'



class ImageDataset(BaseDataset):
    def __init__(self, root, transform=None, mode="RGB"):
        super(ImageDataset, self).__init__()
        self.transform = transform
        self.mode = mode
        self.data = []
        for  class_name in os.listdir(root):
            class_path = os.path.join(root, class_name)
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                self.data.append((img_path, int(class_name)))
        

    def __getitem__(self, idx):
        input, label = self.data[idx]
        input = Image.open(input).convert(self.mode)
        if self.transform:
            input = self.transform(input)
        return input, torch.LongTensor([label])
    
    



