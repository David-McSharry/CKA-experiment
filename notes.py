# %%

import json
from datetime import datetime

import lovely_tensors as lt
import matplotlib.pyplot as plt
import torch
import wandb
from torchvision import datasets, transforms
from utils.utils import display_encoded_samples
from modules.conv_autoencoder import ConvAutoencoder
from train.trainers import train_model_A
from metrics.CKA import CKA_function

device = torch.device("cuda")

lt.monkey_patch()

with open("config.json", "r") as f:
    config = json.load(f)

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

# Download and load the full training images
trainset = datasets.MNIST("./data/", download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=config["batch_size"], num_workers=config["num_workers"]
)

tests_model = ConvAutoencoder(config)

# tests_model.get_full_Hilbert_rep_batch(next(iter(trainloader))[0])
with torch.no_grad():
    x = tests_model.get_full_Hilbert_rep(trainloader)
# tests_model.get_full_Hilbert_rep(trainloader)

# %%
