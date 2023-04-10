import os
import glob
import time
import json
import yaml
import wandb
import pickle
import random
import argparse
import numpy as np
import pandas as pd
import matplotlib.cm as cm
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import lightning.pytorch as pl
import torch
import torch.nn as nn
from utils import Config, set_seed
from data import load_abm_data, ABMDataProcessor
from src import FeedForward, DilatedCNN
from lightning.pytorch.loggers import WandbLogger

def set_model(config, device):
    if config.model == "feedforward":
        return FeedForward(config).to(device)
    elif config.model == "dilatedcnn":
        return DilatedCNN(num_stats = config.num_feat_cols).to(device)
    

def plot_trajectories(model, config, X_test, y_test, R0_list):
    pred_len = config.t_steps - config.context_len
    PLOTS_PATH = config.save_load_path+config.exp_version+"/plots"
    temporal_mse_df = []
    print('\n')
    print("_______________________________________Making Plots_______________________________________")
    print('\n')
    X_test = torch.from_numpy(X_test)
    y_test = torch.from_numpy(y_test)
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
            #print("MODEL INPUT SHAPE", input.shape)
            output = model(input)
            squeeze_output = output.squeeze()
            squeeze_output = squeeze_output.cpu().detach().numpy()
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
        # print("Actuals Shape", actuals.shape)
        # print("Preds Shape", preds.shape)
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
                ax[0, idx].imshow(actuals[i, sdx, :, :], vmin=0, vmax=1)
                ax[0, idx].title.set_text("Act_{}".format(i+1))
                
                ax[1, idx].imshow(preds[i, sdx, :, :], vmin=0, vmax=1)
                ax[1, idx].title.set_text("Pred_{}".format(i+1))

                ax[0, idx].axis('off')
                ax[1, idx].axis('off')
                idx += 1
            
            
            fig.suptitle("R0_{}_{}".format(Exp_Param, stat))
            # plt.show()
            plt.savefig(PLOTS_PATH + "/R0_{}/{}_predictions.png".format(Exp_Param, stat))
            plt.close()
            if (config.use_wandb):
                wandb.log({"{}_predictions.png".format(stat): wandb.Image(PLOTS_PATH + "/R0_{}/{}_predictions.png".format(Exp_Param, stat))})
                # print("\n")
            subplot_rows = 3
            subplot_cols = 3
            fig, ax = plt.subplots(subplot_rows, subplot_cols, figsize=(5, 7))
            # idx = 0

            days = iter(range(9, 95, 10))
            for j in range(0, 3):
                for k in range(0, 3):
                    i = next(days)
                    ax[j, k].imshow(actuals[i, sdx, :, :], vmin=0, vmax=2)
                    ax[j, k].title.set_text("Act_Day_{}".format(i+1))

                    ax[j, k].axis('off')

            fig.suptitle("R0_{}_{}".format(Exp_Param, stat))
            plt.tight_layout()
            # plt.show()
            plt.savefig(PLOTS_PATH + "/R0_{}/{}_actuals.png".format(Exp_Param, stat))
            plt.close()
            if (config.use_wandb):
                wandb.log({"{}_actuals.png".format(stat): wandb.Image(PLOTS_PATH + "/R0_{}/{}_actuals.png".format(Exp_Param, stat))})
            # print("\n")

            subplot_rows = 3
            subplot_cols = 3
            fig, ax = plt.subplots(subplot_rows, subplot_cols, figsize=(5, 7))
            # idx = 0

            days = iter(range(9, 95, 10))
            for j in range(0, 3):
                for k in range(0, 3):
                    i = next(days)
                    ax[j, k].imshow(preds[i, sdx, :, :], vmin=0, vmax=2)
                    ax[j, k].title.set_text("Pred_Day_{}".format(i+1))

                    ax[j, k].axis('off')

            fig.suptitle("R0_{}_{}".format(Exp_Param, stat))
            plt.tight_layout()
            # plt.show()
            plt.tight_layout()
            plt.savefig(PLOTS_PATH + "/R0_{}/{}_preds.png".format(Exp_Param, stat))
            plt.close()
            if (config.use_wandb):
                wandb.log({"{}_preds.png".format(stat): wandb.Image(PLOTS_PATH + "/R0_{}/{}_preds.png".format(Exp_Param, stat))})
            # print("\n")

            colors = iter(cm.rainbow(np.linspace(0, 1, len(actuals))))
            plt.figure(figsize=(5, 5))
            for udx in range(len(actuals)):
                plt.tight_layout()
                #print("SHAPE CHECKK", actuals.shape, preds.shape, actuals[udx, sdx, :, :].shape, preds[udx, sdx, :, :].shape)
                plt.scatter(x=actuals[udx, sdx, :, :]*1000, y=preds[udx, sdx, :, :]*1000, color=next(colors), label='day_'+str(udx))

            plt.xlabel("Actuals")
            plt.ylabel("Predictions")
            fig.suptitle("R0_{}_{}".format(Exp_Param, stat))
            plt.xlim([0, 1000])
            plt.ylim([0, 1000])
            # plt.show()
            plt.tight_layout()
            plt.savefig(PLOTS_PATH + "/R0_{}/{}_predictions_vs_actuals_scatter.png".format(Exp_Param, stat))
            plt.close()
            if (config.use_wandb):
                wandb.log({"{}_predictions_vs_actuals_scatter.png".format(stat): wandb.Image(PLOTS_PATH + "/R0_{}/{}_predictions_vs_actuals_scatter.png".format(Exp_Param, stat))})
            # print("\n\n\n")

            plt.xlabel("days")
            plt.ylabel(stat)

            # mse[stat].append(mean_squared_error(preds[:, :, :, sdx], actuals[:, :, :, sdx]))

            # ensure predictions are cumulative
            if (sdx == 0):
                temporal_preds = np.sum(preds, axis=(2, 3))[:, sdx]
                temporal_preds_rolled = np.roll(temporal_preds, 1)
                temporal_preds_rolled[0] = 0
                temporal_preds_rolled = temporal_preds - temporal_preds_rolled 
                temporal_preds_rolled = np.where(temporal_preds_rolled < 0, 0, temporal_preds_rolled).cumsum()

                temporal_actuals = np.sum(actuals, axis=(2, 3))[:, sdx]
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
                plt.plot(np.sum(actuals, axis=(2, 3))[:, sdx], label='actual')
                plt.plot(np.sum(preds, axis=(2, 3))[:, sdx], label='prediction')
                #plt.plot(np.sum(preds - actuals, axis=(1, 2))[:, sdx], label='difference')
                #temporal_mse[stat].append(mean_squared_error(np.sum(preds, axis=(1, 2))[:, sdx], np.sum(actuals, axis=(1, 2))[:, sdx]))
                temporal_mse[stat].append(np.sum(preds, axis=(2, 3))[:, sdx] - np.sum(actuals, axis=(2, 3))[:, sdx])
                # if args["CI"] == True:
                # plt.fill_between(np.arange(0,95), (95*(np.mean(preds, axis=(1, 2)) + 2 * np.std(preds, axis=(1, 2))))[:, sdx], (95*(np.mean(preds, axis=(1, 2)) - 2 * np.std(preds, axis=(1, 2))))[:, sdx], color="red", alpha=0.1, label=f"95% confidence interval")    

            plt.legend()
            plt.tight_layout()
            plt.savefig(PLOTS_PATH + "/R0_{}/{}_temporal_plot.png".format(Exp_Param, stat))
            plt.close()
            if (config.use_wandb):
                wandb.log({"{}_temporal_plot.png".format(stat): wandb.Image(PLOTS_PATH + "/R0_{}/{}_temporal_plot.png".format(Exp_Param, stat))})
                
        temporal_mse["R0"] = Exp_Param
        temporal_mse["index"] = kdx
        temporal_mse_df.append(temporal_mse)

    return temporal_mse_df    


if __name__ == "__main__":  
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="neural-agent-based-modeling/config.yaml",
                        help="config file path")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        if (config["use_wandb"]):
            wandb.init(
                name=config["exp_version"], project=config["wandb_project"], config=config, save_code=True)

        config = Config(**config)


    set_seed(config.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading data")
    X_train, y_train, X_val, y_val, X_test, y_test, X_test_R0s = load_abm_data(config.data_path, config)
    # print("Data loaded. Length =", len(data))
    print(X_test.shape, y_test.shape, X_test_R0s.shape)
    print('\n')
    # data_processor = ABMDataProcessor(config)
    # train_dataloader = data_processor.build_dataloaders(X_train, y_train, config.train_batch_size)
    # val_dataloader =  data_processor.build_dataloaders(X_val, y_val, config.val_batch_size)

    model = set_model(config, device)

    try:
        model = model.load_from_checkpoint(config.save_load_path + config.exp_version + "/model_weights.pth")
        print("Weights loaded.", '\n')
    except: 
        print("No model found at {}".format(config.save_load_path + config.exp_version + "/model_weights.pth"), '\n')
        pass
        
    print("________________________________________________________")

    temporal_mse_df = plot_trajectories(config, model, X_test, y_test, X_test_R0s)
    temporal_mse_df = pd.DataFrame(temporal_mse_df)
    temporal_mse_df["cumulative_positive_tested"] = temporal_mse_df["cumulative_positive_tested"].apply(lambda x: x[0])
    temporal_mse_df["current_hospitalized"] = temporal_mse_df["current_hospitalized"].apply(lambda x: x[0])
    temporal_mse_df["current_asymptomatic_free"] = temporal_mse_df["current_asymptomatic_free"].apply(lambda x: x[0])
    print("________________Temporal MSE________________")
    print("cumulative_positive_tested", temporal_mse_df["cumulative_positive_tested"].mean())
    print("current_hospitalized", temporal_mse_df["current_hospitalized"].mean())
    print("current_asymptomatic_free", temporal_mse_df["current_asymptomatic_free"].mean())
    print("___________________________________________")
    temporal_mse_df
    print("_________________________________________________________")
    temporal_mse_df.to_csv(config.save_load_path + config.exp_version + '/temporal_mse_df.csv', index=False)
    temporal_mse_df_table = wandb.Table(dataframe=temporal_mse_df)
    table_artifact = wandb.Artifact("temporal_mse", type="dataset")
    table_artifact.add(temporal_mse_df_table, "iris_table")
    # We will also log the raw csv file within an artifact to preserve our data
    table_artifact.add_file(config.save_load_path + config.exp_version + '/temporal_mse_df.csv')
    wandb.log_artifact(table_artifact)