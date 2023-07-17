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

# Halve the dataset to fit memory constraints.
trainset, _ = torch.utils.data.random_split(
    trainset, [len(trainset) // 2, len(trainset) // 2]
)

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=config["batch_size"], num_workers=config["num_workers"]
)

a1_trainset, a2_trainset = torch.utils.data.random_split(
    trainset, [len(trainset) // 2, len(trainset) // 2]
)
a1_trainloader = torch.utils.data.DataLoader(
    a1_trainset, batch_size=config["batch_size"]
)
a2_trainloader = torch.utils.data.DataLoader(
    a2_trainset, batch_size=config["batch_size"]
)

# Download and load the full test images
testset = datasets.MNIST("./data/", download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=config["batch_size"])


# %% [markdown]
# # Load model

loaded = torch.load("model_a1.pth")
model_a1 = ConvAutoencoder(config)
model_a1.load_state_dict(loaded["model_state_dict"])

# %% [markdown]
# # Visualize features

def visualize_features(model, input_image):
    with torch.no_grad():
        x = input_image.to(model.device)
        # Pass through the first convolution layer
        x = model.encoder[0](x)
        x = model.encoder[1](x)  # Pass through the ReLU layer

        # x is now a feature map with shape (B, conv_dim, 14, 14)
        # Let's visualize the features for the first image in the batch
        features = x[0].cpu().numpy()

        # We can plot the features for each filter as an image
        plt.figure(figsize=(10, 10))
        for i in range(features.shape[0]):
            plt.subplot(4, 4, i+1)
            plt.imshow(features[i], cmap='gray')
        plt.show()

# Now, you can feed an input_image and visualize the features
visualize_features(model_a1, next(iter(trainloader))[0])

# %%
