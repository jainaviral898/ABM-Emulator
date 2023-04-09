import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

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

class DilatedCNN(nn.Module):
    def __init__(self, num_stats=3, input_shape=torch.rand(4, 5, 5, 10, 10).shape):
        super(DilatedCNN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=input_shape[1],
            out_channels=256,
            kernel_size=(5, 5),
            stride=1,
            padding=0,
            dilation=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=256,
            out_channels=128,
            kernel_size=(3, 3),
            stride=1,
            padding=0,
            dilation=2
        )
        self.conv3 = nn.Conv2d(
            in_channels=128,
            out_channels=64,
            kernel_size=(1, 1),
            stride=1,
            padding=0,
            dilation=4
        )
        self.fc1 = nn.Linear(
            in_features=1280,
            out_features=1000
        )
        self.fc2 = nn.Linear(
            in_features=1000,
            out_features=input_shape[4] * input_shape[3] * num_stats
        )
        self.num_stats = num_stats
        self.input_shape = input_shape

    def forward(self, x):
        y = []
        x = x.reshape(-1, x.shape[0], x.shape[2], x.shape[3], x.shape[4])
        #print("here", x.shape)
        for xi in x:
            xi = xi.float()
            xi = F.relu(self.conv1(xi))
            xi = F.relu(self.conv2(xi))
            xi = F.relu(self.conv3(xi))
            y.append(xi)
        x = torch.stack(y)
        #print(x.shape)
        x = x.permute(1, 0, 2, 3, 4)
        x = x.reshape(x.shape[0], -1)
        #print(x.shape)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        #print("fc2", x.shape)
        x = x.view(-1, 1, self.num_stats, self.input_shape[4], self.input_shape[3])
        return x

class UNet(nn.Module):

    def __init__(self):
        super().__init__()
                
        self.dconv_down1 = double_conv(5, 12)
        self.dconv_down2 = double_conv(12, 24)
        self.dconv_down3 = double_conv(24, 24)
        
        self.maxpool = nn.MaxPool2d(2)

        self.upsample1 = nn.Upsample(scale_factor=2.5, mode='bilinear', align_corners=True)        
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        self.dconv_up3 = double_conv(48, 24)
        self.dconv_up2 = double_conv(24, 12)
        
        self.conv_last = nn.Conv2d(12, 3, 1)
        
        
    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)
        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)        
        conv3 = self.dconv_down3(x)
        x = self.upsample1(x)  
        #print(x.shape, conv2.shape)      
        x = torch.cat([x, conv2], dim=1)
        #print(x.shape)
        x = self.dconv_up3(x)
        #print("before upsample", x.shape)
        x = self.upsample2(x)
        x = nn.Conv2d(24, 12, 3, padding='same')(x)      
        #print(x.shape, conv1.shape)
        x = torch.cat([x, conv1], dim=1)
        #print(x.shape)
        x = self.dconv_up2(x)
        out = self.conv_last(x)
        #print(out.shape)
        return out