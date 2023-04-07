import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torchvision.transforms import Compose, Normalize, ToTensor, Resize


class ABMDataProcessor():
    def __init__(self, config):
        self.config = config

    def collate_fn(self, items):
        batch = {
            "trajectory": torch.stack([x['trajectory'] for x in items], dim=0), 
            "next_step": torch.stack([x['next_step'] for x in items], dim=0)
        }
        return batch
    
    # def fn(self, trajectory_list):
    #     row = [] 

    #     for i in range(len(trajectory_list)):
    #         # for j in range(0, 95, 5): 0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95
    #         for j in range(0, len(trajectory_list[i]) - self.config.context_len, self.config.context_len):
    #             data = {}
    #             # data = {"trajectory": [], "next_step": [], "R0": []}
    #             # data["trajectory"].append(trajectory_list[i][j : (j + self.config.context_len), :])
    #             data["trajectory"] = torch.Tensor(trajectory_list[i][j:(j + self.config.context_len), :, :, :])
    #             data["next_step"] = torch.Tensor(trajectory_list[i][(j + self.config.context_len), :, :, :self.config.num_feat_cols])
    #             # data["R0"] = torch.from_numpy(np.array(trajectory_list[i][j + self.config.context_len,0,0,4]))
    #             # can add time step like R0

    #             row.append(data)

    #     return row

    def fn(self, X, y):
        row = []

        for i in range(len(X)):
            data = {}
            data["trajectory"] = torch.Tensor(X[i])
            data["next_step"] = torch.Tensor(y[i])
            row.append(data)
        
        return row
    

    def build_dataloaders(self, X, y, B):
        # trajectories_train, trajectories_test = train_test_split(trajectories, test_size=self.config.test_fraction, random_state=self.config.seed, shuffle=True)
        # trajectories_train, trajectories_val = train_test_split(trajectories_train, test_size=self.config.val_fraction/self.config.train_fraction, random_state=self.config.seed, shuffle=True)

        # print("train, val, test sizes")
        # print(len(trajectories_train), len(trajectories_val), len(trajectories_test))

        # print("train data mean, std")
        self.channel_mean = [np.mean(np.stack(X, axis=0)[:,:,:,:,i]) for i in range(X.shape[-1])]
        print("mean list len", len(self.channel_mean))

        self.channel_std = [np.std(np.stack(X, axis=0)[:,:,:,:,i]) for i in range(X.shape[-1])]
        print("std list len", len(self.channel_std))

        # self.channel_mean = [np.mean(np.stack(trajectories_train, axis=0)[:, :, :, :, i]) for i in range(trajectories_train[0].shape[-1])]
        # self.channel_std = [np.std(np.stack(trajectories_train, axis=0)[:, :, :, :, i]) for i in range(trajectories_train[0].shape[-1])]
        # print(self.channel_mean, self.channel_std)

        data_transform = Compose(
            [
                ToTensor(),
                Normalize(self.channel_mean, self.channel_std),
            ]
        )

        data = self.fn(X, y)

        print("data sample")
        print(f"{data[0]['trajectory'].shape=}")
        # [5, 10, 10, 5]
        # {'trajectory': torch.Size([4, 100, 5, 10, 10])}
        # test data -> list of len 988, one element of list is a dict with keys: trajectory, next_step, R0, of shape (5, 10, 10, 5)
        # make X_test, y_test of shape (988, 5, 10, 10, 5), (988, 1, 10, 10, 3)
        # and R0_list of shape (988, 1)

        # X_test = []
        # y_test = []
        # R0_list = []
        # for batch in test_data:
        #     X_test.append(batch["trajectory"])
        #     y_test.append(batch["next_step"])
        #     R0_list.append(batch["R0"])

        # X_test = torch.stack(X_test, dim=0)
        # y_test = torch.stack(y_test, dim=0)
        # R0_list = torch.stack(R0_list, dim=0)

        dataloader = torch.utils.data.DataLoader(data, batch_size=B, collate_fn=self.collate_fn)
        # val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=self.config.val_batch_size, collate_fn=self.collate_fn)
        # test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=self.config.val_batch_size, collate_fn=self.collate_fn)

        print("train_dataloader sample")
        for batch in dataloader:
            print({key: batch[key].shape for key in batch.keys()})
            break

        return dataloader
