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
            "next_step": torch.stack([x['next_step'] for x in items], dim=0),
            "R0": torch.stack([x['R0'] for x in items], dim=0)
        }

        return batch
    
    def fn(self, trajectory_list):
        row = [] 

        for i in range(len(trajectory_list)):
            for j in range(0, len(trajectory_list[i]) - self.config.context_len, self.config.context_len):
                data = {}
                # data = {"trajectory": [], "next_step": [], "R0": []}
                # data["trajectory"].append(trajectory_list[i][j : (j + self.config.context_len), :])
                data["trajectory"] = torch.Tensor(trajectory_list[i][j:(j + self.config.context_len), :, :, :])
                data["next_step"] = torch.Tensor(trajectory_list[i][(j + self.config.context_len), :, :, :self.config.num_feat_cols])
                data["R0"] = torch.from_numpy(np.array(trajectory_list[i][j + self.config.context_len,4,0,0]))
                # can add time step like R0

                row.append(data)

        return row
    

    def build_dataloaders(self, trajectories):
        trajectories_train, trajectories_test = train_test_split(trajectories, test_size=self.config.test_fraction, random_state=self.config.seed, shuffle=True)
        trajectories_train, trajectories_val = train_test_split(trajectories_train, test_size=self.config.val_fraction/self.config.train_fraction, random_state=self.config.seed, shuffle=True)

        print("train, val, test sizes")
        print(len(trajectories_train), len(trajectories_val), len(trajectories_test))

        print("train data mean, std")
        self.channel_mean = [np.mean(np.stack(trajectories_train, axis=0)[:, :, :, :, i]) for i in range(trajectories_train[0].shape[-1])]
        self.channel_std = [np.std(np.stack(trajectories_train, axis=0)[:, :, :, :, i]) for i in range(trajectories_train[0].shape[-1])]
        print(self.channel_mean, self.channel_std)

        data_transform = Compose(
            [
                ToTensor(),
                Normalize(self.channel_mean, self.channel_std),
            ]
        )

        train_data = self.fn(trajectories_train)
        val_data = self.fn(trajectories_val)
        test_data = self.fn(trajectories_test)

        print("train_data sample")
        print(f"{train_data[0]['trajectory'].shape=}")
        # [5, 10, 10, 5]
        # {'trajectory': torch.Size([4, 100, 5, 10, 10])}

                

        train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=self.config.train_batch_size, collate_fn=self.collate_fn)
        val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=self.config.val_batch_size, collate_fn=self.collate_fn)
        test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=self.config.val_batch_size, collate_fn=self.collate_fn)

        print("train_dataloader sample")
        for batch in train_dataloader:
            print({key: batch[key].shape for key in batch.keys()})
            break

        return train_dataloader, val_dataloader, test_dataloader

# invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
#                                                      std = [ 1/0.229, 1/0.224, 1/0.225 ]),
#                                 transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
#                                                      std = [ 1., 1., 1. ]),
#                                ])

# inv_tensor = invTrans(inp_tensor)