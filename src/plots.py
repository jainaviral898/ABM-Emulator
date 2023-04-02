import os
import wandb
import numpy as np
from tqdm import tqdm

from PIL import Image
from matplotlib import cm
import matplotlib.pyplot as plt
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError  

# pred_len = args["SIM_DAYS"]-args["LAG"] 
pred_len = 95
PLOTS_PATH = "/content/plots"

# def plot_trajectories(config, x_grid, t_grid, actual_trajectory_tensor, predicted_trajectory_tensor, save_dir, save_plots_every_x_trajectories, save_single_step_plots=False):
#     os.makedirs(save_dir, exist_ok=True)
#     mesh_x, mesh_t = np.meshgrid(x_grid, t_grid)

#     mean_squared_error = MeanSquaredError()

#     columns = ["id", "MSE", "RMSE", "actual_3D", "prediction_3D", "actual_2D", "prediction_2D"]
#     plots_table = wandb.Table(columns=columns)
#     for idx in tqdm(range(0, actual_trajectory_tensor.shape[0], save_plots_every_x_trajectories)):
#         actual = actual_trajectory_tensor[idx, :, :, :].squeeze()
#         predicted = predicted_trajectory_tensor[idx, :, :, :].squeeze()

#         actual_3D, prediction_3D, actual_2D, prediction_2D = make_trajectory_plots(x_grid, t_grid, mesh_x, mesh_t, actual, predicted, save_dir, idx, save_single_step_plots)

#         mse = mean_squared_error(actual, predicted)
#         rmse = np.sqrt(mse)

#         plots_table.add_data(idx, mse, rmse, wandb.Image(actual_3D), wandb.Image(prediction_3D), wandb.Image(actual_2D), wandb.Image(prediction_2D))

#     return plots_table


def make_plots(actuals, preds, config):
    
    for sdx, stat in enumerate(config.stats):
        subplot_rows = 2
        subplot_cols = pred_len//5
        fig, ax = plt.subplots(subplot_rows, subplot_cols, figsize=(20, 3))
        # change to idx = 0 (?)
        idx = -1
        for i in range(0, pred_len, 5):
            ax[0, idx].imshow(actuals[i, :, :, sdx], vmin=0, vmax=1)
            ax[0, idx].title.set_text("Act_{}".format(i+1))
            
            ax[1, idx].imshow(preds[i, :, :, sdx], vmin=0, vmax=1)
            ax[1, idx].title.set_text("Pred_{}".format(i+1))

            ax[0, idx].axis('off')
            ax[1, idx].axis('off')
            idx += 1

        fig.suptitle("{}_{}_{}".format(args["EXP_TYPE"], Exp_Param, stat))
        # plt.show()
        plt.savefig(PLOTS_PATH + "/{}_{}/{}_predictions.png".format(args["EXP_TYPE"], Exp_Param, stat))
        plt.close()
        # print("\n")

        subplot_rows = 3
        subplot_cols = 3
        fig, ax = plt.subplots(subplot_rows, subplot_cols, figsize=(5, 7))
        # idx = 0

        days = iter(range(9, pred_len, 10))
        for j in range(0, 3):
            for k in range(0, 3):
                i = next(days)
                ax[j, k].imshow(actuals[i, :, :, sdx], vmin=0, vmax=2)
                ax[j, k].title.set_text("Act_Day_{}".format(i+1))

                ax[j, k].axis('off')

        fig.suptitle("{}_{}_{}".format(args["EXP_TYPE"], Exp_Param, stat))
        plt.tight_layout()
        # plt.show()
        plt.savefig(PLOTS_PATH + "/{}_{}/{}_actuals.png".format(args["EXP_TYPE"], Exp_Param, stat))
        plt.close()
        # print("\n")

        subplot_rows = 3
        subplot_cols = 3
        fig, ax = plt.subplots(subplot_rows, subplot_cols, figsize=(5, 7))
        # idx = 0

        days = iter(range(9, pred_len, 10))
        for j in range(0, 3):
            for k in range(0, 3):
                i = next(days)
                ax[j, k].imshow(preds[i, :, :, sdx], vmin=0, vmax=2)
                ax[j, k].title.set_text("Pred_Day_{}".format(i+1))

                ax[j, k].axis('off')

        fig.suptitle("{}_{}_{}".format(args["EXP_TYPE"], Exp_Param, stat))
        plt.tight_layout()
        # plt.show()
        plt.tight_layout()
        plt.savefig(PLOTS_PATH + "/{}_{}/{}_preds.png".format(args["EXP_TYPE"], Exp_Param, stat))
        plt.close()
        # print("\n")

        colors = iter(cm.rainbow(np.linspace(0, 1, len(actuals))))
        plt.figure(figsize=(5, 5))
        for udx in range(len(actuals)):
            plt.tight_layout()
            plt.scatter(x=actuals[udx, :, :, sdx]*args["BLOCK_COUNT"], y=preds[udx, :, :, sdx]*args["BLOCK_COUNT"], color=next(colors), label='day_'+str(udx))

        plt.xlabel("Actuals")
        plt.ylabel("Predictions")
        fig.suptitle("{}_{}_{}".format(args["EXP_TYPE"], Exp_Param, stat))
        plt.xlim([0, 1000])
        plt.ylim([0, 1000])
        # plt.show()
        plt.tight_layout()
        plt.savefig(PLOTS_PATH + "/{}_{}/{}_predictions_vs_actuals_scatter.png".format(args["EXP_TYPE"], Exp_Param, stat))
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
            plt.plot(temporal_preds_rolled - temporal_actuals_rolled, label='difference')
            #temporal_mse[stat].append(mean_squared_error(temporal_preds_rolled, temporal_actuals_rolled))
            temporal_mse[stat].append(temporal_preds_rolled - temporal_actuals_rolled)
            if args["CI"] == True:
                plt.fill_between(np.arange(0,95), (95*(np.mean(preds, axis=(1, 2)) + 2 * np.std(preds, axis=(1, 2))))[:, sdx], (95*(np.mean(preds, axis=(1, 2)) - 2 * np.std(preds, axis=(1, 2))))[:, sdx], color="red", alpha=0.1, label=f"95% confidence interval")    

        else:
            plt.plot(np.sum(actuals, axis=(1, 2))[:, sdx], label='actual')
            plt.plot(np.sum(preds, axis=(1, 2))[:, sdx], label='prediction')
            plt.plot(np.sum(preds - actuals, axis=(1, 2))[:, sdx], label='difference')
            #temporal_mse[stat].append(mean_squared_error(np.sum(preds, axis=(1, 2))[:, sdx], np.sum(actuals, axis=(1, 2))[:, sdx]))
            temporal_mse[stat].append(np.sum(preds, axis=(1, 2))[:, sdx] - np.sum(actuals, axis=(1, 2))[:, sdx])
            if args["CI"] == True:
              plt.fill_between(np.arange(0,95), (95*(np.mean(preds, axis=(1, 2)) + 2 * np.std(preds, axis=(1, 2))))[:, sdx], (95*(np.mean(preds, axis=(1, 2)) - 2 * np.std(preds, axis=(1, 2))))[:, sdx], color="red", alpha=0.1, label=f"95% confidence interval")    


        plt.legend()
        plt.tight_layout()
        plt.savefig(PLOTS_PATH + "/{}_{}/{}_temporal_plot.png".format(args["EXP_TYPE"], Exp_Param, stat))
        plt.close()