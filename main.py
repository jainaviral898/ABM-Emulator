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
from src import FeedForward, SingleStepTrainer

def set_params(config):
    
    # loss = "MSE" or "CrossEntropy" or "BCE"
    # optimizer = "Adam" or "SGD" or "Adagrad" or "Adadelta"
    # scheduler = "multiplicative" or "step" or "exponential" or "plateau"
    # model = "FeedForward" or "DilatedCNN" or "UNet"
    
    if config.loss_fn == "MSE": 
        loss_fn = torch.nn.MSELoss()
    elif config.loss_fn == "CrossEntropy":
        loss_fn = torch.nn.CrossEntropyLoss()
    elif config.loss_fn == "BCE":
        loss_fn = torch.nn.BCELoss()

    if config.optimizer == "Adam":
        optimizer =  torch.optim.Adam()
    elif config.optimizer == "SGD":
        optimizer = torch.optim.SGD()
    elif config.optimizer == "Adagrad":
        optimizer = torch.optim.Adagrad()
    elif config.optimizer == "Adadelta":
        optimizer = torch.optim.Adadelta()

    if config.scheduler == "multiplicative": 
        lmbda = lambda epoch: 0.65 ** epoch
        scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lmbda)
    elif config.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
    elif config.scheduler == "exponential":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5)
    elif config.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    if config.model == "FeedForward":
        model = FeedForward(config)
    elif config.model == "DilatedCNN":
        model = DilatedCNN(config)
    elif config.model == "UNet":
        model = UNet(config)

    return loss_fn, optimizer, scheduler, model   


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

    print("loading data")
    data = load_abm_data(config.data_path)

    data_processor = ABMDataProcessor(config)
    train_dataloader, val_dataloader, test_dataloader = data_processor.build_dataloaders(data)

    loss_fn, optimizer, scheduler, model = set_params(config)
    # model = FeedForward(config)
    model.to(device)
    
    print("test model on sample input")
    out = model(torch.rand(4, config.context_len, 5, 10, 10))
    print(out.shape)

    # loss_fn = torch.nn.MSELoss()

    # optimizer = torch.optim.Adam(
    #     model.parameters(), lr=config.learning_rate)

    num_training_steps = config.train_epochs * len(train_dataloader)

    # scheduler = get_scheduler(
    #     "linear",
    #     optimizer=optimizer,
    #     num_warmup_steps=0,
    #     num_training_steps=num_training_steps,
    # )
    
    trainer = SingleStepTrainer(model, loss_fn, optimizer, scheduler, config, device)
    if (config.train):
        trainer.train(train_dataloader)

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