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
        loss = 0
        context = context.permute(0, 1, 4, 2, 3)
        # print("Context Shape =", context.shape) = [4, 5, 5, 10, 10]
        prediction = self.model(context)
        prediction = torch.squeeze(prediction, dim=1)
        target = batch["next_step"] # actual_trajectory[:, time_step, :self.config.num_feat_cols, :, :].unsqueeze(dim=1).float().to(self.device)  
        target = target.permute(0, 3, 1, 2)
        batch_loss = loss_function(prediction, target)
        loss = batch_loss
        
        self.log("train_loss", loss)

        if (self.config.use_wandb):
            wandb.log({"train_batch_loss": loss})

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

    def validation_step(self, batch, batch_idx):
        x = batch['trajectory']
        y = batch['next_step']
        x = x.permute(0, 1, 4, 2, 3)
        scores = self.model(x)
        scores = torch.squeeze(scores)
        y = y.permute(0, 3, 1, 2)
        loss = loss_function(scores, y)
        self.log("val_loss", loss)
        return loss


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
            context = context.permute(0, 1, 4, 2, 3)
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
            target = batch["next_step"] # actual_trajectory[:, time_step, :self.config.num_feat_cols, :, :].unsqueeze(dim=1).float().to(self.device)  
            target = target.permute(0, 3, 1, 2)  
            # [B, 1, num_feat_cols, x_dim, y_dim]
            print(f"{target.shape=}")

            # update loss
            batch_loss = loss_function(target, prediction)
            loss += batch_loss

            # update context tensor
            # [B, context_len, num_feat_cols + num_stat_cols, x_dim, y_dim] # (constant size)
            context = context.permute(0, 1, 4, 2, 3)
            prediction = torch.squeeze(prediction)
            print("context shape = ", context.shape, "context_concat shape = ", context[:,-1,:self.config.num_feat_cols,:,:].shape, "prediction shape = ", prediction.shape)
            #[4, 5, 5, 10, 10], [4, 3, 10, 10], [4, 3, 10, 10]
            context = torch.cat((context[:,-1,:self.config.num_feat_cols,:,:], prediction), dim=1)
            # context = [4, 6, 10, 10]
            print(actual_trajectory[:, time_step, self.config.num_feat_cols:, :, :].shape)
            context[:, -1, self.config.num_feat_cols:, :, :] = actual_trajectory[:, time_step, self.config.num_feat_cols:, :, :]
            
            # update predicted_trajectory
            predicted_trajectory = torch.cat((predicted_trajectory, prediction), dim=1)
        
        # plots.plot_predictions(actual_trajectory, predicted_trajectory, self.config)
        
        return {"loss" : loss, "actual_trajectory" : actual_trajectory[:, :, :self.config.num_feat_cols, :, :], "predicted_trajectory" : predicted_trajectory}

    def test_step(self, batch):
        context = batch['trajectory']
        context = context.permute(0, 1, 4, 2, 3)
        actual_trajectory = batch['next_step']
        predicted_trajectory = self.model(context)
        return actual_trajectory, predicted_trajectory
        
    def test(self, dataloader):
        print("Dataloader Shape", len(dataloader))
        actual_trajectory_list = []
        predicted_trajectory_list = []
        for batch in tqdm(dataloader):
            actual_trajectory, predicted_trajectory = self.test_step(batch)
            actual_trajectory_list.append(actual_trajectory)
            predicted_trajectory_list.append(predicted_trajectory.cpu())

        actual_trajectory_tensor = torch.cat(actual_trajectory_list, dim=0)  
        # [B, t_step, num_feat_cols, x_dim, y_dim]
        print(f"Test Step ___________ {actual_trajectory_tensor.shape=}")

        predicted_trajectory_tensor = torch.cat(predicted_trajectory_list, dim=0)  
        # [B, t_step, num_feat_cols, x_dim, y_dim]
        print(f"Test Step ___________ {predicted_trajectory_tensor.shape=}")

        return actual_trajectory_tensor, predicted_trajectory_tensor


    def plot_trajectories(self, config, actual_trajectory_tensor, predicted_trajectory_tensor):
        PLOTS_PATH = "/content/plots"
        Exp_Param = 1.2
        
        os.makedirs(PLOTS_PATH + "/R0_{}".format(Exp_Param), exist_ok=True)
        # predicted_trajectory_tensor shape = [988, 1, 3, 10, 10]
        predicted_trajectory_tensor = predicted_trajectory_tensor.permute(0,1,3,4,2)
        predicted_trajectory_tensor = torch.squeeze(predicted_trajectory_tensor)
        print('\n', actual_trajectory_tensor.shape, predicted_trajectory_tensor.shape)
        # torch.Size([988, 10, 10, 3]) torch.Size([988, 10, 10, 3])
        # for idx in tqdm(range(0, actual_trajectory_tensor.shape[0])):
        actuals = actual_trajectory_tensor.detach().numpy()
        preds = predicted_trajectory_tensor.detach().numpy()

        for sdx, stat in enumerate(config.stats):
            subplot_rows = 2
            subplot_cols = 95//5
            fig, ax = plt.subplots(subplot_rows, subplot_cols, figsize=(20, 3))
            # change to idx = 0 (?)
            idx = -1
            for i in range(0, 95, 5):
                #print(actuals.shape)
                ax[0, idx].imshow(actuals[i, :, :, sdx], vmin=0, vmax=1)
                ax[0, idx].title.set_text("Act_{}".format(i+1))
                
                ax[1, idx].imshow(preds[i, :, :, sdx], vmin=0, vmax=1)
                ax[1, idx].title.set_text("Pred_{}".format(i+1))

                ax[0, idx].axis('off')
                ax[1, idx].axis('off')
                idx += 1
            
            
            fig.suptitle("R0_{}_{}".format(Exp_Param, stat))
            # plt.show()
            plt.savefig(PLOTS_PATH + "/R0_{}/{}_predictions.png".format(Exp_Param, stat))
            plt.show()
            plt.close()
                # print("\n")
            
            #mse = mean_squared_error(actuals, preds)
            


            #plots_table.add_data(idx, mse, rmse, wandb.Image(actual_3D), wandb.Image(prediction_3D), wandb.Image(actual_2D), wandb.Image(prediction_2D))

        
    
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

    # def make_trajectory_plots(self, x_grid, t_grid, mesh_x, mesh_t, actual_trajectory, predicted_trajectory, save_dir, traj_idx):
    #     make_3D_plot_for_1D_trajectory(actual_trajectory.cpu().squeeze().numpy(), x_grid, t_grid, traj_idx, 
    #                                    save_path=self.config.save_load_path + save_dir + "/{}-actual.png".format(traj_idx))
        
    #     actual_3D = Image.open(
    #         self.config.save_load_path + save_dir + "/{}-actual.png".format(traj_idx))

    #     return actual_3D
    

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)
