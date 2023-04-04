import os
import wandb
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from torchmetrics import MeanAbsolutePercentageError
import lightning.pytorch as pl
from src import plots


import torch

def loss_function(outputs, targets):
  # MSE + i/L (penalty)
  mse = torch.nn.MSELoss()
  return mse(outputs, targets)

def loss_fn2(outputs, targets, i, L):
  # loss = (1/n) * ∑(|y - ŷ| / y) * (1 + sign(y - ŷ)) + i/L (penalty)
  # mape * sign + i/L
  mape = MeanAbsolutePercentageError()
  mape_ = mape(outputs, targets) * (1 + torch.sign(targets - outputs))
  return torch.add(mape_, torch.divide(i, L))

class SingleStepTrainer(pl.LightningModule):
    def __init__(self, model, config):
        super(SingleStepTrainer, self).__init__()
        self.model = model
        self.config = config
        self.train_epochs = self.config.train_epochs

    def training_step(self, batch):
        context = batch["trajectory"]  
        # [B, t_steps, num_feat_cols + num_stat_cols, x_dim, y_dim]
        # print(f"{actual_trajectory.shape=}") 

        # [B, context_len, num_feat_cols, x_dim, y_dim] # (t_steps will be added incrementally)
        # predicted_trajectory = batch["trajectory"][:, :self.config.context_len, :self.config.num_feat_cols, :, :].float().to(self.device)
        # print(f"{predicted_trajectory.shape=}")

        # [B, context_len, num_feat_cols + num_stat_cols, x_dim, y_dim] # (constant size)
        # context = batch["trajectory"][:, :self.config.context_len, :, :, :].float().to(self.device)
        # print(f"{context.shape=}")

        # rolling loss
        loss = 0

        # for time_step in range(self.config.context_len, self.config.t_steps):
        #     # get prediction for the current time_step
        prediction = self.model(context)
        # if (len(prediction.shape) == 2):
        #     prediction = prediction.unsqueeze(
        #         dim=1).unsqueeze(dim=1)  # [B, 1, 1, 1024]
        # elif(len(prediction.shape) == 1):
        #     prediction = prediction.unsqueeze(dim=0).unsqueeze(
        #         dim=0).unsqueeze(dim=0)  # [B, 1, 1, 1024]
        # else:
        #     raise AssertionError
        # [B, 1, num_feat_cols, x_dim, y_dim]
        # print(f"{prediction.shape=}")

        # get target for the current time_step
        target = batch["next_step"] # actual_trajectory[:, time_step, :self.config.num_feat_cols, :, :].unsqueeze(dim=1).float().to(self.device)  
        # [B, 1, num_feat_cols, x_dim, y_dim]
        # print(f"{target.shape=}")

        # update loss
        #batch_loss = self.loss_fn(target, prediction)
        print(prediction.shape, target.shape)
        batch_loss = loss_function(prediction, target)
        loss = batch_loss
        # self.optimizer.zero_grad()
        # batch_loss.backward()
        # self.optimizer.step()

        # remove prediction from the computation graph
        # prediction = prediction.detach()

        # # update context tensor
        # # [B, context_len, num_feat_cols + num_stat_cols, x_dim, y_dim] # (constant size)
        # context[:, :, :self.config.num_feat_cols, :, :] = torch.cat((context[:, 1:, :self.config.num_feat_cols, :, :], prediction), dim=1)
        # context[:, -1, self.config.num_feat_cols:, :, :] = actual_trajectory[:, time_step, self.config.num_feat_cols:, :, :]

        
        if (self.config.use_wandb):
            wandb.log({"batch_loss": loss})

        # self.scheduler.step(batch_loss)

        return loss

    # def train(self, dataloader):
    #     self.model.train()
    #     for epoch in tqdm(range(self.train_epochs)):
    #         for batch in dataloader:
    #             epoch_loss = self.training_step(batch)

    #             if (self.config.use_wandb):
    #                 wandb.log({"learning_rate"
    #                           : self.scheduler.get_last_lr()[0]})

    #         if (self.config.use_wandb):
    #             wandb.log({"epoch_loss"
    #                       : epoch_loss.item()/len(dataloader)})

    # def validation_step(self, batch, batch_idx):
    #     x, y = batch
        
        

    def predict_step(self, batch, batch_idx):
        # with torch.no_grad():
        actual_trajectory = batch["trajectory"]  
        # [B, t_steps, num_feat_cols + num_stat_cols, x_dim, y_dim]
        print(f"{actual_trajectory.shape=}") 

        # [B, context_len, num_feat_cols, x_dim, y_dim] # (t_steps will be added incrementally)
        predicted_trajectory = batch["trajectory"][:, :self.config.context_len, :self.config.num_feat_cols, :, :].float().to(self.device)
        print(f"{predicted_trajectory.shape=}")

        # [B, context_len, num_feat_cols + num_stat_cols, x_dim, y_dim] # (constant size)
        context = batch["trajectory"][:, :self.config.context_len, :, :, :].float().to(self.device)

        # rolling loss
        loss = 0

        for time_step in range(self.config.context_len, self.config.t_steps):
            # get prediction for the current time_step
            prediction = self.model(context)
            # if (len(prediction.shape) == 2):
            #     prediction = prediction.unsqueeze(
            #         dim=1).unsqueeze(dim=1)  # [B, 1, 1, 1024]
            # elif(len(prediction.shape) == 1):
            #     prediction = prediction.unsqueeze(dim=0).unsqueeze(
            #         dim=0).unsqueeze(dim=0)  # [B, 1, 1, 1024]
            # else:
            #     raise AssertionError
            # [B, 1, num_feat_cols, x_dim, y_dim]
            print(f"{prediction.shape=}")

            # get target for the current time_step
            target = actual_trajectory[:, time_step, :self.config.num_feat_cols, :, :].unsqueeze(dim=1).float().to(self.device)  
            # [B, 1, num_feat_cols, x_dim, y_dim]
            print(f"{target.shape=}")

            # update loss
            batch_loss = self.loss_fn(target, prediction)
            loss += batch_loss

            # update context tensor
            # [B, context_len, num_feat_cols + num_stat_cols, x_dim, y_dim] # (constant size)
            context = torch.cat((context[:, 1:, :self.config.num_feat_cols, :, :], prediction), dim=1)
            context[:, -1, self.config.num_feat_cols:, :, :] = actual_trajectory[:, time_step, self.config.num_feat_cols:, :, :]

            # update predicted_trajectory
            predicted_trajectory = torch.cat((predicted_trajectory, prediction), dim=1)
        
        # plots.plot_predictions(actual_trajectory, predicted_trajectory, self.config)
        
        return {"loss" : loss, "actual_trajectory" : actual_trajectory[:, :, :self.config.num_feat_cols, :, :], "predicted_trajectory" : predicted_trajectory}

    # def test(self, dataloader):
    #     self.model.eval()

    #     actual_trajectory_list = []
    #     predicted_trajectory_list = []
    #     for batch in tqdm(dataloader):
    #         actual_trajectory, predicted_trajectory = self.test_step(batch)
    #         actual_trajectory_list.append(actual_trajectory)
    #         predicted_trajectory_list.append(predicted_trajectory.cpu())

    #     actual_trajectory_tensor = torch.cat(actual_trajectory_list, dim=0)  
    #     # [B, t_step, num_feat_cols, x_dim, y_dim]
    #     print(f"{actual_trajectory_tensor.shape=}")

    #     predicted_trajectory_tensor = torch.cat(predicted_trajectory_list, dim=0)  
    #     # [B, t_step, num_feat_cols, x_dim, y_dim]
    #     print(f"{predicted_trajectory_tensor.shape=}")

    #     return actual_trajectory_tensor, predicted_trajectory_tensor



    # def calculate_metrics(self, dataloader, x_grid, t_grid, exp_name):
    #     print("predicting trajectories")
    #     actual_trajectory_tensor, predicted_trajectory_tensor = self.test(dataloader)

    #     print("calculating metrics and making plots")
    #     # if (self.config.use_wandb):
    #     #     columns = ["id", "MSE", "RMSE", "actual_3D",
    #     #                "prediction_3D", "actual_2D", "prediction_2D"]
    #     #     plots_table = wandb.Table(columns=columns)

    #     save_dir = "/{}".format(exp_name)
    #     os.makedirs(self.config.save_load_path + save_dir, exist_ok=True)
    #     mesh_x, mesh_t = np.meshgrid(x_grid, t_grid)

    #     mse_list = []
    #     rmse_list = []
    #     for idx in tqdm(range(actual_trajectory_tensor.shape[0])):
    #         actual = actual_trajectory_tensor[idx, :, :, :, :].squeeze()
    #         predicted = predicted_trajectory_tensor[idx, :, :, :, :].squeeze()

    #         mse_list.append(mean_squared_error(actual, predicted))
    #         rmse_list.append(mean_squared_error(
    #             actual, predicted, squared=False))

    #         # actual_3D, prediction_3D, actual_2D, prediction_2D = self.make_trajectory_plots(
    #         #     x_grid, t_grid, mesh_x, mesh_t, actual, predicted, save_dir, idx)

    #         # columns=["id", "MSE", "RMSE", "actual_3D", "prediction_3D", "actual_2D", "prediction_2D"]
    #         # if (self.config.use_wandb):
    #         #     plots_table.add_data(idx, mse_list[-1], rmse_list[-1], wandb.Image(actual_3D), wandb.Image(
    #         #         prediction_3D), wandb.Image(actual_2D), wandb.Image(prediction_2D))

    #     # if (self.config.use_wandb):
    #     #     wandb.log({exp_name: plots_table})

    #     return np.mean(mse_list), np.mean(rmse_list)

    # def make_trajectory_plots(self, x_grid, t_grid, mesh_x, mesh_t, actual_trajectory, predicted_trajectory, save_dir, traj_idx):
    #     make_3D_plot_for_1D_trajectory(actual_trajectory.cpu().squeeze().numpy(), x_grid, t_grid, traj_idx, 
    #                                    save_path=self.config.save_load_path + save_dir + "/{}-actual.png".format(traj_idx))
    #     actual_3D = Image.open(
    #         self.config.save_load_path + save_dir + "/{}-actual.png".format(traj_idx))

    #     make_2D_plot_for_1D_trajectory(actual_trajectory, x_grid, t_grid, traj_idx, 
    #                                    save_path=self.config.save_load_path + save_dir + "/{}-2D-actual.png".format(traj_idx))
    #     actual_2D = Image.open(
    #         self.config.save_load_path + save_dir + "/{}-2D-actual.png".format(traj_idx))

    #     make_3D_plot_for_1D_trajectory(actual_trajectory.cpu().squeeze().numpy(), x_grid, t_grid, traj_idx, 
    #                                    save_path=self.config.save_load_path + save_dir + "/{}-predictions.png".format(traj_idx))
    #     prediction_3D = Image.open(
    #         self.config.save_load_path + save_dir + "/{}-predictions.png".format(traj_idx))

    #     make_2D_plot_for_1D_trajectory(predicted_trajectory, x_grid, t_grid, traj_idx, 
    #                                    save_path=self.config.save_load_path + save_dir + "/{}-2D-predictions.png".format(traj_idx))
    #     prediction_2D = Image.open(
    #         self.config.save_load_path + save_dir + "/{}-2D-predictions.png".format(traj_idx))

    #     return actual_3D, prediction_3D, actual_2D, prediction_2D
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)
