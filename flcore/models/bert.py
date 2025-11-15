from transformers import BertModel, RobertaModel
import torch
import torch.nn as nn

class BertSequenceClassifier(nn.Module):
    '''
    Sentence Level classification using Bert for encoding sentence
    '''
    def __init__(self, hidden_size=768, classes=4):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.classifier = nn.Linear(hidden_size, classes)

    def forward(self, input_ids, attention_mask):
        '''
        input_ids = [N x L], containing sequence of ids of words after tokenization
        attention_mask = [N x L], mask for attention

        N = batch size
        L = maximum sentence length
        '''
        output = self.bert(input_ids, attention_mask)
        sentence_embedding = output.pooler_output # 1st token followed by linear layer and tanh
        logits = self.classifier(sentence_embedding)
        return logits

class RobertaSequenceClassifier(nn.Module):
    '''
    Sentence Level classification using Roberta for encoding sentence
    '''
    def __init__(self, hidden_size=768, classes=4):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.classifier = nn.Linear(hidden_size, classes)

    def forward(self, input_ids, attention_mask):
        '''
        input_ids = [N x L], containing sequence of ids of words after tokenization
        attention_mask = [N x L], mask for attention

        N = batch size
        L = maximum sentence length
        '''
        output = self.roberta(input_ids, attention_mask)
        sentence_embedding = output.pooler_output # 1st token followed by linear layer and tanh
        logits = self.classifier(sentence_embedding)
        return logits