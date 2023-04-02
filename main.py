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

import torch
import torch.nn as nn
# from transformers import get_scheduler


from utils import Config, set_seed
from data import load_abm_data, ABMDataProcessor
from src import FeedForward, SingleStepTrainer, DilatedCNN, make_plots

def set_model(config, device):
    if config.model == "feedforward":
        return FeedForward(config).to(device)
    elif config.model == "dilatedcnn":
        return DilatedCNN(num_stats = config.num_feat_cols).to(device)

def set_optimizer(config, model):
    if config.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    elif config.optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), lr=config.learning_rate)
    
def set_scheduler(config, optimizer):
    if config.scheduler == "multiplicative":
        lmbda = lambda epoch: 0.65 ** epoch
        return torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lmbda)
    elif config.scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    elif config.scheduler == "decay":
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5)
    
def set_loss_fn(config):
    if config.loss_fn == "mse":
        return torch.nn.MSELoss()



if __name__ == "__main__":  
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="neural-agent-based-modeling/config.yaml",
                        help="config file path")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        if (config["use_wandb"]):
            wandb.init(
                name=config["wandb_path"], project=config["wandb_project"], config=config, save_code=True)
        config = Config(**config)

    set_seed(config.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading data")
    data = load_abm_data(config.data_path)
    print("Data loaded. Length =", len(data))

    data_processor = ABMDataProcessor(config)
    train_dataloader, val_dataloader, test_dataloader = data_processor.build_dataloaders(data)

    # model = FeedForward(config)
    # model.to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    # lmbda = lambda epoch: 0.65 ** epoch
    # scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lmbda)
    # loss_fn = torch.nn.MSELoss()

    model     = set_model(config, device)
    optimizer = set_optimizer(config, model)
    scheduler = set_scheduler(config, optimizer)
    loss_fn   = set_loss_fn(config)
    
    print("Testing model on sample input")
    out = model(torch.rand(4, config.context_len, 5, 10, 10).to(device))
    print("Model Output Shape", out.shape)
    print("________________________________________________________")

    num_training_steps = config.train_epochs * len(train_dataloader)

    trainer = SingleStepTrainer(model, loss_fn, optimizer, scheduler, config, device)
    if (config.train):
        trainer.train(train_dataloader)
        print("Training complete.")

    # maybe save model and optimizer
    if (config.save_model_optimizer):
        print("saving model, optimizer, and scheduler at {}/model_optimizer_scheduler.pt".format(config.save_load_path))
        os.makedirs("{}/".format(config.save_load_path), exist_ok=True)
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }, "{}/model_optimizer_scheduler.pt".format(config.save_load_path))

    # maybe load model and optimizer
    if (config.load_model_optimizer):
        print("loading model and optimizer from {}/model_optimizer_scheduler.pt".format(config.save_load_path))
        checkpoint = torch.load(
            "{}/model_optimizer_scheduler.pt".format(config.save_load_path))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if(config.make_plots):
        outputs = trainer.predict()
        make_plots(outputs["actual_trajectory"], outputs["predicted_trajectory"], config)