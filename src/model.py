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

class DilatedCNN(torch.nn.Module):
    def __init__(self):
        super(DilatedCNN, self).__init__()
        
        self.fclayers = nn.Sequential(
        nn.Conv2d(in_channels=5, out_channels=256, kernel_size = 5, stride = 1, padding= 'valid', dilation = 1),
        nn.Conv2d(in_channels=256, out_channels=128, kernel_size = 3, stride = 1, padding= 'valid', dilation = 2),
        nn.Conv2d(in_channels=128, out_channels=64, kernel_size = 1, stride = 1, padding= 'valid', dilation = 4),
        nn.Flatten(start_dim=0),
        nn.Linear(1280, 1000),
        nn.ReLU(),
        nn.Linear(1000, 300),
        nn.ReLU(),
        )
    def forward(self, x):
        print("the shape is", x.shape)
        x = self.fclayers(x)
        x = torch.reshape(x, (1, 3, 10, 10))
        return x

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding='same'),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding='same'),
        nn.ReLU(inplace=True)
    )   

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