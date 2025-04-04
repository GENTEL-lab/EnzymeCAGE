import copy
import sys

import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_batch

from .base import BaseModel


class Baseline(BaseModel):
    def __init__(self, esm_model, hidden_dims=[3328, 2048], device='cpu', sigmoid_readout=False):
        super(Baseline, self).__init__()
        # hidden_dim = 3328
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.LeakyReLU(),
            nn.Linear(hidden_dims[1], 1)
        )
        self.model_device = device
        self.sigmoid_readout = sigmoid_readout
    
    def forward(self, data):
        esm_embedding, _ = to_dense_batch(data.esm_feature, data.esm_feature_batch)    
        reaction_feature, _ = to_dense_batch(data.reaction_feature, data.reaction_feature_batch)
        output = self.mlp(torch.cat([esm_embedding, reaction_feature], dim=-1))
        if self.sigmoid_readout:
            output = torch.sigmoid(output)
        return output.squeeze(-1)
    
    @torch.no_grad()
    def predict(self, dataloader):
        preds = []
        for batch in dataloader:
            pred = self.forward(batch.to(self.model_device))
            preds.append(pred)
        pred = torch.cat(preds)
        return pred
    


