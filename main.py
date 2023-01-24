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

from utils import Config, set_seed
from data import load_abm_data, ABMDataProcessor


if __name__ == "__main__":  
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml",
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

    
