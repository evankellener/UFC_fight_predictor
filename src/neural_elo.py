import torch
import torch.nn as nn

class NeuralElo(nn.Module):
    def __init__(self, num_players):
        super().__init__()
        self.embeddings = nn.Embedding(num_players, 1)

    def forward(self, p1, p2):
        r1 = self.embeddings(p1).squeeze()
        r2 = self.embeddings(p2).squeeze()
        return torch.sigmoid(r1 - r2)