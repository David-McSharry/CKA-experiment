# %%

import json
from datetime import datetime

import lovely_tensors as lt
import matplotlib.pyplot as plt
import torch
import wandb
from torchvision import datasets, transforms
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

# Set torch seed
torch.manual_seed(1)

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


# %%

models = [
    ("A1", "model_a1.pt"),
    ("B1", "model_b1.pt"),
    ("B2", "model_b2.pt"),
    ("B3", "model_b3.pt"),
    ("B4", "model_b4.pt"),
    ("B5", "model_b5.pt"),
    ("B6", "model_b6.pt"),
    ("B7", "model_b7.pt"),
    ("B8", "model_b8.pt"),
]

import numpy as np

SAMPLE_NUM = 4

samples = next(iter(testloader))[0][:SAMPLE_NUM].to(device)


# %%
# Plot the samples in a row

fig, axs = plt.subplots(1, len(samples), figsize=(8, 2))

for i, sample in enumerate(samples):
    axs[i].imshow(sample.cpu().numpy().squeeze(), cmap="gray")
    axs[i].set_title(f"Sample {i + 1}")


# %%
# Now, for each model, let's visualize the latents.

LATENT_DIM = 12

latents = np.full((len(models) * len(samples), LATENT_DIM), np.nan)

with torch.no_grad():
    for model_index, (label, path) in enumerate(models):
        model = ConvAutoencoder(config)
        path = f"models/{path}"
        model.load_state_dict(torch.load(path))
        model.to(device)

        for i, sample in enumerate(samples):
            model_latents = model.encoder(sample.unsqueeze(0)).cpu().numpy().squeeze()
            latents[model_index * len(samples) + i] = model_latents


# %%
# Let's plot all the latents for the first sample, by model.

# Two columns: one for the sample, one for the latents.
# The latents subplot will have one row per model, labelled.

comparison_grid = np.full((len(models) * len(samples), LATENT_DIM), np.nan)


plt.figure(figsize=(10, 10))

for model_index, (label, path) in enumerate(models):
    for sample_index, sample in enumerate(samples):
        # if model_index == 0:
        #     continue

        comparison_grid[model_index * len(samples) + sample_index] = latents[
            model_index * len(samples) + sample_index
        ]
        # Add y labels
        plt.text(
            -1,
            model_index * len(samples) + sample_index,
            f"{sample_index + 1}",
            verticalalignment="center",
            horizontalalignment="right",
        )

    # Add group labels
    plt.text(
        -2,
        model_index * len(samples) + len(samples) / 2,
        f"Model {label}",
        verticalalignment="center",
        horizontalalignment="right",
        fontweight="bold",
    )


plt.imshow(comparison_grid, cmap="Blues")
plt.title("Latents for each model")
plt.xlabel("Latent dimension")

# Remove y ticks
plt.yticks([])

# Add model names as tick labels, and a space between each model
for i in range(len(models)):
    # horizontal line
    plt.axhline(
        i * len(samples) - 0.5,
        color="black",
        linewidth=1,
    )

    plt.xticks(np.arange(LATENT_DIM), np.arange(LATENT_DIM) + 1)

# Let's add a cmap
plt.colorbar()

plt.show()


# %%
