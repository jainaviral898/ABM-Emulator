import os
import wandb
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from torchmetrics import MeanAbsolutePercentageError
import lightning.pytorch as pl
from src import plots
import torch

def loss_function(outputs, targets):
  # MSE + i/L (penalty)
  mse = torch.nn.MSELoss()
  return mse(outputs, targets)

def loss_fn_mape(outputs, targets):
    mape = MeanAbsolutePercentageError()
    mape_ = mape(outputs, targets) * (1 + torch.sign(targets - outputs))
    return mape_

def loss_fn_penalty(outputs, targets, i, L):
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
        self.loss_fn = loss_function

    def training_step(self, batch):
        context = batch["trajectory"]  
        loss = 0
        context = context.permute(0, 1, 4, 2, 3)
        # print("Context Shape =", context.shape) = [4, 5, 5, 10, 10]
        prediction = self.model(context)
        prediction = torch.squeeze(prediction, dim=1)
        target = batch["next_step"] # actual_trajectory[:, time_step, :self.config.num_feat_cols, :, :].unsqueeze(dim=1).float().to(self.device)  
        target = target.permute(0, 3, 1, 2)
        batch_loss = self.loss_fn(prediction, target)
        loss = batch_loss
        
        self.log("train_loss", loss)

        if (self.config.use_wandb):
            wandb.log({"train_batch_loss": loss})

        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['trajectory']
        y = batch['next_step']
        x = x.permute(0, 1, 4, 2, 3)
        scores = self.model(x)
        scores = torch.squeeze(scores)
        y = y.permute(0, 3, 1, 2)
        loss = self.loss_fn(scores, y)
        self.log("val_loss", loss)
        return loss

    def predict_step(self, batch, batch_idx):
        # with torch.no_grad():
        context = batch["trajectory"]  
        context = context.permute(0, 1, 4, 2, 3)
        # context.shape - [4, 5, 5, 10, 10]
        R0 = batch["R0"]
        actual_trajectory = batch["next_step"]
        # [B, t_steps, num_feat_cols + num_stat_cols, x_dim, y_dim]
        print(f"{actual_trajectory.shape=}") 
        #       actual_trajectory.shape=torch.Size([4, 10, 10, 3])

        # [B, context_len, num_feat_cols, x_dim, y_dim] # (t_steps will be added incrementally)
        #predicted_trajectory = batch["trajectory"][:, :self.config.context_len, :self.config.num_feat_cols, :, :].float().to(self.device)
        predicted_trajectory = self.model(context)
        print(f"{predicted_trajectory.shape=}")

        # rolling loss
        loss = 0

        for time_step in range(self.config.context_len, self.config.t_steps):
            # get prediction for the current time_step
            prediction = self.model(context)
            
            print(f"{prediction.shape=}")

            # get target for the current time_step
            target = batch["next_step"] # actual_trajectory[:, time_step, :self.config.num_feat_cols, :, :].unsqueeze(dim=1).float().to(self.device)  
            target = target.permute(0, 3, 1, 2)  
            # [B, 1, num_feat_cols, x_dim, y_dim]
            print(f"{target.shape=}")

            # update loss
            batch_loss = self.loss_fn(target, prediction)
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
        
        return {"loss" : loss, "actual_trajectory" : actual_trajectory[:, :, :self.config.num_feat_cols, :, :], "predicted_trajectory" : predicted_trajectory, "R0" : R0}

    def test_step(self, batch):
        context = batch['trajectory']
        R0 = batch['R0']
        context = context.permute(0, 1, 4, 2, 3)
        actual_trajectory = batch['next_step']
        predicted_trajectory = self.model(context)
        return R0, actual_trajectory, predicted_trajectory
        
    def test(self, dataloader):
        print("Dataloader Shape", len(dataloader))
        actual_trajectory_list = []
        predicted_trajectory_list = []
        R0_list = []
        loss_list = []
        for batch in tqdm(dataloader):
            loss, actual_trajectory, predicted_trajectory, R0 = self.predict_step(batch)
            actual_trajectory_list.append(actual_trajectory)
            predicted_trajectory_list.append(predicted_trajectory.cpu())
            R0_list.append(R0)
            loss_list.append(loss)

        actual_trajectory_tensor = torch.cat(actual_trajectory_list, dim=0)  
        # [B, t_step, num_feat_cols, x_dim, y_dim]
        print(f"Test Step ___________ {actual_trajectory_tensor.shape=}")

        predicted_trajectory_tensor = torch.cat(predicted_trajectory_list, dim=0)  
        # [B, t_step, num_feat_cols, x_dim, y_dim]
        print(f"Test Step ___________ {predicted_trajectory_tensor.shape=}")

        return loss_list, actual_trajectory_tensor, predicted_trajectory_tensor, R0_list


    def plot_trajectories(self, config, X_test, y_test, R0_list):
        pred_len = 95
        PLOTS_PATH = "/content/plots"
        print('\n')
        print("_______________________________________Making Plots_______________________________________")
        print('\n')
        X_test = X_test.permute((0,1,4,2,3))
        print("X_test shape", X_test.shape)
        for kdx, i in tqdm(enumerate(range(0, len(X_test), pred_len)), total=len(range(0, len(X_test), pred_len))):
            Exp_Param = float(R0_list[kdx])

            os.makedirs(PLOTS_PATH + "/R0_{}".format(Exp_Param), exist_ok=True)
            
            input = X_test.squeeze()[i:i+1] # right exclusive means we only take one series
            # input = np.expand_dims(input, axis=0)
            # input = input.permute(0,1,4,2,3)
            #print("INPUT SHAPE after", input.shape, type(input))
            preds = []
            actuals = []
            
            
            for j in range(0, pred_len):
              output = self.model(input)
              squeeze_output = output.squeeze()
              squeeze_output = squeeze_output.detach().numpy()
              preds.append(squeeze_output)  
              actuals.append(y_test[i + j - 1].squeeze())
              # 1 = Horizon
              input = np.roll(input, -1, axis=1)
              input[0, -1, :config.num_feat_cols, :, :] = squeeze_output
              input[0, -1, config.num_feat_cols:, :, :] = X_test[i+j-1, -1, config.num_feat_cols:, :, :]
              input = torch.from_numpy(input)

            #print(f"{actuals[0].shape =}")
            actuals = [x.permute(2, 0, 1) for x in actuals]
            preds = np.stack(preds)
            actuals = np.stack(actuals)
            mse = {key: [] for key in config.stats}
            temporal_mse = {key: [] for key in config.stats}

            for sdx, stat in enumerate(config.stats):
                subplot_rows = 2
                subplot_cols = 95//5
                fig, ax = plt.subplots(subplot_rows, subplot_cols, figsize=(20, 3))
                # change to idx = 0 (?)
                idx = -1
                for i in range(0, 95, 5):
                    # print(f"{actuals.shape =}")
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
                plt.close()
                    # print("\n")
                subplot_rows = 3
                subplot_cols = 3
                fig, ax = plt.subplots(subplot_rows, subplot_cols, figsize=(5, 7))
                # idx = 0

                days = iter(range(9, 95, 10))
                for j in range(0, 3):
                    for k in range(0, 3):
                        i = next(days)
                        ax[j, k].imshow(actuals[i, :, :, sdx], vmin=0, vmax=2)
                        ax[j, k].title.set_text("Act_Day_{}".format(i+1))

                        ax[j, k].axis('off')

                fig.suptitle("R0_{}_{}".format(Exp_Param, stat))
                plt.tight_layout()
                # plt.show()
                plt.savefig(PLOTS_PATH + "/R0_{}/{}_actuals.png".format(Exp_Param, stat))
                plt.close()
                # print("\n")

                subplot_rows = 3
                subplot_cols = 3
                fig, ax = plt.subplots(subplot_rows, subplot_cols, figsize=(5, 7))
                # idx = 0

                days = iter(range(9, 95, 10))
                for j in range(0, 3):
                    for k in range(0, 3):
                        i = next(days)
                        ax[j, k].imshow(preds[i, :, :, sdx], vmin=0, vmax=2)
                        ax[j, k].title.set_text("Pred_Day_{}".format(i+1))

                        ax[j, k].axis('off')

                fig.suptitle("R0_{}_{}".format(Exp_Param, stat))
                plt.tight_layout()
                # plt.show()
                plt.tight_layout()
                plt.savefig(PLOTS_PATH + "/R0_{}/{}_preds.png".format(Exp_Param, stat))
                plt.close()
                # print("\n")

                colors = iter(cm.rainbow(np.linspace(0, 1, len(actuals))))
                plt.figure(figsize=(5, 5))
                for udx in range(len(actuals)):
                    plt.tight_layout()
                    #print("SHAPE CHECKK", actuals.shape, preds.shape, actuals[udx, :, :, sdx].shape, preds[udx, :, :, sdx].shape)
                    plt.scatter(x=actuals[udx, :, :, sdx]*1000, y=preds[udx, :, :, sdx]*1000, color=next(colors), label='day_'+str(udx))

                plt.xlabel("Actuals")
                plt.ylabel("Predictions")
                fig.suptitle("R0_{}_{}".format(Exp_Param, stat))
                plt.xlim([0, 1000])
                plt.ylim([0, 1000])
                # plt.show()
                plt.tight_layout()
                plt.savefig(PLOTS_PATH + "/R0_{}/{}_predictions_vs_actuals_scatter.png".format(Exp_Param, stat))
                plt.close()
                # print("\n\n\n")

                plt.xlabel("days")
                plt.ylabel(stat)

                # mse[stat].append(mean_squared_error(preds[:, :, :, sdx], actuals[:, :, :, sdx]))

                # ensure predictions are cumulative
                if (sdx == 0):
                    temporal_preds = np.sum(preds, axis=(1, 2))[:, sdx]
                    temporal_preds_rolled = np.roll(temporal_preds, 1)
                    temporal_preds_rolled[0] = 0
                    temporal_preds_rolled = temporal_preds - temporal_preds_rolled 
                    temporal_preds_rolled = np.where(temporal_preds_rolled < 0, 0, temporal_preds_rolled).cumsum()

                    temporal_actuals = np.sum(actuals, axis=(1, 2))[:, sdx]
                    temporal_actuals_rolled = np.roll(temporal_actuals, 1)
                    temporal_actuals_rolled[0] = 0
                    temporal_actuals_rolled = temporal_actuals - temporal_actuals_rolled 
                    temporal_actuals_rolled = np.where(temporal_actuals_rolled < 0, 0, temporal_actuals_rolled).cumsum()

                    plt.plot(temporal_actuals_rolled, label='actual')
                    plt.plot(temporal_preds_rolled, label='prediction')
                    #plt.plot(temporal_preds_rolled - temporal_actuals_rolled, label='difference')
                    #temporal_mse[stat].append(mean_squared_error(temporal_preds_rolled, temporal_actuals_rolled))
                    temporal_mse[stat].append(temporal_preds_rolled - temporal_actuals_rolled)
                    # if args["CI"] == True:
                    #     plt.fill_between(np.arange(0,95), (95*(np.mean(preds, axis=(1, 2)) + 2 * np.std(preds, axis=(1, 2))))[:, sdx], (95*(np.mean(preds, axis=(1, 2)) - 2 * np.std(preds, axis=(1, 2))))[:, sdx], color="red", alpha=0.1, label=f"95% confidence interval")    

                else:
                    plt.plot(np.sum(actuals, axis=(1, 2))[:, sdx], label='actual')
                    plt.plot(np.sum(preds, axis=(1, 2))[:, sdx], label='prediction')
                    #plt.plot(np.sum(preds - actuals, axis=(1, 2))[:, sdx], label='difference')
                    #temporal_mse[stat].append(mean_squared_error(np.sum(preds, axis=(1, 2))[:, sdx], np.sum(actuals, axis=(1, 2))[:, sdx]))
                    temporal_mse[stat].append(np.sum(preds, axis=(1, 2))[:, sdx] - np.sum(actuals, axis=(1, 2))[:, sdx])
                    # if args["CI"] == True:
                    # plt.fill_between(np.arange(0,95), (95*(np.mean(preds, axis=(1, 2)) + 2 * np.std(preds, axis=(1, 2))))[:, sdx], (95*(np.mean(preds, axis=(1, 2)) - 2 * np.std(preds, axis=(1, 2))))[:, sdx], color="red", alpha=0.1, label=f"95% confidence interval")    

                plt.legend()
                plt.tight_layout()
                plt.savefig(PLOTS_PATH + "/R0_{}/{}_temporal_plot.png".format(Exp_Param, stat))
                plt.close()


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)