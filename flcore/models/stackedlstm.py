
import torch
from flcore.models.model_utils import BaseModel



class StackedLSTM(BaseModel):
    def __init__(self, num_classes, embedding_size, num_embeddings, hidden_size, num_layers, is_seq2seq, dropout=0.1):
        super(StackedLSTM, self).__init__()
        self.is_seq2seq = is_seq2seq
        self.num_hiddens = hidden_size
        self.num_classes = num_classes
        self.num_embeddings = num_embeddings
        self.embedding_size = embedding_size
        self.dropout = dropout
        self.num_layers = num_layers
        
        self.features = torch.nn.Sequential(
            torch.nn.Embedding(num_embeddings=self.num_embeddings, embedding_dim=self.embedding_size),
            torch.nn.LSTM(
                input_size=self.embedding_size,
                hidden_size=self.num_hiddens,
                num_layers=self.num_layers,
                batch_first=True,
                dropout=self.dropout,
                bias=True
            )
        )
        self.classifier = torch.nn.Linear(self.num_hiddens, self.num_classes, bias=True)
    
    def extract_first_element(self, x):
        return x[0]

    def forward(self, x):
        x = self.features(x)
        x = self.extract_first_element(x)
        x = self.classifier(x if self.is_seq2seq else x[:, -1, :])
        return x
