import math
import torch

from flcore.models.model_utils import BaseModel


#############################
# Positional Encoding Layer #
#############################
class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, dropout, max_len=10000):
        super().__init__()
        self.d_model = d_model
        self.dropout = torch.nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x * math.sqrt(self.d_model)
        x = x + self.pe[:x.size(0)]
        x = self.dropout(x)
        return x



class StackedTransformer(BaseModel):
    def __init__(self, num_classes, embedding_size, num_embeddings, hidden_size, seq_len, num_layers, is_seq2seq, dropout=0.1):
        super(StackedTransformer, self).__init__()
        self.is_seq2seq = is_seq2seq
        self.num_hiddens = hidden_size
        self.num_classes = num_classes
        self.num_embeddings = num_embeddings
        self.embedding_size = embedding_size
        self.seq_len = seq_len
        self.dropout = dropout
        self.num_layers = num_layers
        
        self.features = torch.nn.Sequential(
            torch.nn.Embedding(num_embeddings=self.num_embeddings, embedding_dim=self.embedding_size),
            PositionalEncoding(self.embedding_size, self.dropout),
            torch.nn.TransformerEncoder(
                torch.nn.TransformerEncoderLayer(self.embedding_size, 16, self.num_hiddens, self.dropout, batch_first=True), 
                self.num_layers
            )
        )
        self.classifier = torch.nn.Linear(self.embedding_size, self.num_classes, bias=True)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x if self.is_seq2seq else x[:, 0, :])
        return x
