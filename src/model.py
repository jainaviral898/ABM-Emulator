import numpy as np

import torch
import torch.nn as nn

class FeedForward(torch.nn.Module):
    def __init__(self, config):
        super(FeedForward, self).__init__()

        self.config = config

        base_dim = (self.config.num_feat_cols + self.config.num_stat_cols) * self.config.context_len * self.config.x_dim * self.config.y_dim
        self.linear1 = nn.Linear(base_dim, base_dim*2)
        self.linear2 = nn.Linear(base_dim*2, base_dim)
        self.linear3 = nn.Linear(base_dim, self.config.num_feat_cols * 1 * self.config.x_dim * self.config.y_dim)

        self.internal_activation = nn.ReLU()

    def forward(self, x):
        x = torch.flatten(x, start_dim=1, end_dim=4)
        x = self.internal_activation(self.linear1(x))
        x = self.internal_activation(self.linear2(x))
        x = self.linear3(x)
        x = torch.reshape(x, (-1, 1, self.config.num_feat_cols, self.config.x_dim, self.config.y_dim))

        return x