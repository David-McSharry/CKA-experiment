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


# %% [markdown]

model_nums = [1, 2, 3, 4]

# for each pair of models, bet the CKA between them
import numpy as np

grid = np.zeros((4, 4))

combinations = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]

with torch.no_grad():
    for model_num_1, model_num_2 in combinations:
        model_1 = ConvAutoencoder(config)
        model_1.load_state_dict(torch.load(f"model_b{model_num_1}.pt"))
        model_2 = ConvAutoencoder(config)
        model_2.load_state_dict(torch.load(f"model_b{model_num_2}.pt"))
        hilbert_vectors_model_1 = model_1.get_full_Hilbert_rep(a1_trainloader)
        hilbert_vectors_model_2 = model_2.get_full_Hilbert_rep(a2_trainloader)
        CKA = CKA_function(hilbert_vectors_model_1, hilbert_vectors_model_2)
        print(f"CKA between model_b{model_num_1} and model_b{model_num_2}: {CKA}")
        grid[model_num_1 - 1, model_num_2 - 1] = CKA
        del model_1
        del model_2
        del hilbert_vectors_model_1
        del hilbert_vectors_model_2


# %%

# make the diagonal values 1

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))

# Create a heatmap
sns.heatmap(grid, vmin=0, vmax=1, annot=True, cmap="YlGnBu")

# Add labels
plt.xlabel("Models")
plt.ylabel("Models")

# Set the labels to be x + 1
plt.xticks(np.arange(4) + 0.5, labels=[1, 2, 3, 4])
plt.yticks(np.arange(4) + 0.5, labels=[1, 2, 3, 4])

# Add title
plt.title("CKA between models")

# Show plot
plt.show()


# %%

# network graph

import networkx as nx

G = nx.Graph()

G.add_nodes_from([1, 2, 3, 4])

G.add_edges_from(
    [
        (1, 2),
        (1, 3),
        (1, 4),
        (2, 3),
        (2, 4),
        (3, 4),
    ]
)

# the distances between the nodes
edge_labels = {
    (1, 2): grid[0, 1],
    (1, 3): grid[0, 2],
    (1, 4): grid[0, 3],
    (2, 3): grid[1, 2],
    (2, 4): grid[1, 3],
    (3, 4): grid[2, 3],
}

# round
edge_labels = {k: round(v, 2) for k, v in edge_labels.items()}

# edge labels
nx.draw_networkx_edge_labels(G, pos=nx.spring_layout(G), edge_labels=edge_labels)

nx.draw(G, with_labels=True, font_weight="bold", node_color="skyblue", node_size=500)

# %%
