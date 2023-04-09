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
from src import FeedForward, SingleStepTrainer, DilatedCNN
from lightning.pytorch.loggers import WandbLogger

def set_model(config, device):
    if config.model == "feedforward":
        return FeedForward(config).to(device)
    elif config.model == "dilatedcnn":
        return DilatedCNN(num_stats = config.num_feat_cols).to(device)


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

    data_processor = ABMDataProcessor(config)
    train_dataloader = data_processor.build_dataloaders(X_train, y_train, config.train_batch_size)
    val_dataloader =  data_processor.build_dataloaders(X_val, y_val, config.val_batch_size)

    model = set_model(config, device)
        
    print("________________________________________________________")
    print("Testing model on sample input")
    out = model(torch.rand(4, config.context_len, 5, 10, 10).to(device))
    print("Model Output Shape", out.shape)
    print("________________________________________________________")
    print('\n')

    num_training_steps = config.train_epochs * len(train_dataloader)

    if (config.load_model):
        try:
            model = model.load_from_checkpoint(config.save_load_path + config.exp_version + 'model_weights.pth')
        except: 
            print("No model found at {}".format(config.save_load_path + config.exp_version), '\n')
            pass

    lightning_mod = SingleStepTrainer(model, config)
    if (config.train):
        trainer = pl.Trainer(accelerator = "gpu", devices = 1, max_epochs = config.train_epochs)       
        trainer.fit(model = lightning_mod, train_dataloaders = train_dataloader)
        print("Training complete.")

    val_loss = trainer.validate(model = lightning_mod, dataloaders=val_dataloader)
    print("Loss on validation set:", val_loss)

    # maybe save model and optimizer
    if (config.save_model):
        print("saving model at {}".format(config.save_load_path + config.exp_version))
        os.makedirs("{}/".format(config.save_load_path + config.exp_version), exist_ok=True)
        # torch.save({
        #     'model_state_dict': model.state_dict(),
        #     'optimizer_state_dict': optimizer.state_dict(),
        #     'scheduler_state_dict': scheduler.state_dict(),
        # }, "{}/model_optimizer_scheduler.pt".format(config.save_load_path + config.exp_version))
        torch.save(model.state_dict(), config.save_load_path + config.exp_version +'model_weights.pth')

    # maybe load model and optimizer
    # if (config.load_model):
    #     # print("loading model and optimizer from {}/model_optimizer_scheduler.pt".format(config.save_load_path + config.exp_version))
    #     # checkpoint = torch.load(
    #     #     "{}/model_optimizer_scheduler.pt".format(config.save_load_path + config.exp_version))
    #     # model.load_state_dict(checkpoint['model_state_dict'])
    #     # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #     model = model.load_from_checkpoint(config.save_load_path + config.exp_version + 'model_weights.pth')


    if(config.make_plots):
        # R0_list, actual_trajectory_tensor, predicted_trajectory_tensor = trainer.test(test_dataloader)
        temporal_mse_df = lightning_mod.plot_trajectories(config, X_test, y_test, X_test_R0s)
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
        temporal_mse_df.to_csv(config.save_load_path + config.exp_version + '/temporal_mse_df.csv', index=False)
        temporal_mse_df_table = wandb.Table(dataframe=temporal_mse_df)
        table_artifact = wandb.Artifact("temporal_mse", type="dataset")
        table_artifact.add(temporal_mse_df_table, "iris_table")
        # We will also log the raw csv file within an artifact to preserve our data
        table_artifact.add_file(config.save_load_path + config.exp_version + '/temporal_mse_df.csv')
        wandb.log_artifact(table_artifact)