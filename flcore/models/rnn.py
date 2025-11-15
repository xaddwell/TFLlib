import torch
import torch.nn as nn

from flcore.models.model_utils import BaseModel


def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


class RNN(BaseModel):
    def __init__(self, embedding_size=8, num_embeddings=80, num_hiddens=256, batch_first=True, num_layers=2):
        super(RNN, self).__init__()
        self.encoder = nn.Embedding(num_embeddings, embedding_dim=embedding_size)
        self.lstm = nn.LSTM(
            input_size=embedding_size, 
            hidden_size=num_hiddens, 
            num_layers=num_layers, 
            batch_first=batch_first
            )
        self.decoder = nn.Linear(num_hiddens, num_embeddings)
        self.init_weights()

    def forward(self, inp):
        inp = self.encoder(inp)
        inp, _ = self.lstm(inp)
        # extract the last state of output for prediction
        hidden = inp[:, -1]
        output = self.decoder(hidden)
        return output

    def init_weights(self):
        init_range = 0.1
        self.encoder.weight.data.uniform_(-init_range, init_range)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init_range, init_range)
