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


# %%

models = [
    ("Model B1", "model_b1.pt"),
    ("Model B2", "model_b2.pt"),
    ("Model B3", "model_b3.pt"),
    ("Model B4", "model_b4.pt"),
    ("Model B5", "model_b5.pt"),
    ("Model B6", "model_b6.pt"),
    ("Model B7", "model_b7.pt"),
    ("Model B8", "model_b8.pt"),
]

# for each pair of models, bet the CKA between them
import numpy as np

grid = np.full((len(models), len(models)), np.nan)


with torch.no_grad():
    for model_num_1, (model_label_1, model_path_1) in enumerate(models):
        model_1 = ConvAutoencoder(config)
        model_1.load_state_dict(torch.load(model_path_1))
        hilbert_vectors_model_1 = model_1.get_full_Hilbert_rep(a1_trainloader)

        for model_num_2, (model_label_2, model_path_2) in enumerate(models):
            if model_num_1 < model_num_2:
                continue

            model_2 = ConvAutoencoder(config)
            model_2.load_state_dict(torch.load(model_path_2))

            hilbert_vectors_model_2 = model_2.get_full_Hilbert_rep(a1_trainloader)

            CKA = CKA_function(hilbert_vectors_model_1, hilbert_vectors_model_2)

            print(f"CKA between {model_label_1} and {model_label_2} is {CKA}")
            grid[model_num_1, model_num_2] = CKA


# %%

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Set the figure size
plt.figure(figsize=(10, 8))

sns.heatmap(grid, vmin=0, vmax=1, annot=True, fmt=".3f", cmap="YlGnBu")
# Create a heatmap
# Add title and axis names
plt.title("CKA between models")
plt.xlabel("Model")
plt.ylabel("Model")

# Add model names as tick labels
tick_labels = [model[0] for model in models]
plt.xticks(np.arange(len(models)) + 0.5, tick_labels, rotation=90)
plt.yticks(np.arange(len(models)) + 0.5, tick_labels, rotation=0)

plt.show()


# %% [markdown]
# Cluster the models based on their CKA values.

# First, we need to convert our grid of CKA values into a distance matrix.
# We can do this by subtracting the CKA values from 1.

# %%


# Replace NaNs with 0s in the grid of CKA values
distance_matrix = grid.copy()
distance_matrix[np.isnan(distance_matrix)] = 0

# Since the CKA distance matrix is symmetric, we can add it to its transpose
distance_matrix = distance_matrix + distance_matrix.T

# The diagonal elements of the distance matrix should be 1, since the distance between a model and itself is 0
np.fill_diagonal(distance_matrix, 1)

# Convert the CKA values to distances by subtracting them from 1
distance_matrix = 1 - distance_matrix


# %% [markdown]
# Let's use k-means clustering to cluster the models based on their CKA values.

from sklearn.cluster import KMeans

n_clusters = 4

# Apply k-means clustering
kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(distance_matrix)

# Get the cluster labels for each object in your grid
cluster_labels = kmeans.labels_

# Now, let's draw a set of `n_clusters` heatmaps, one for each cluster.
# Each heatmap will show the CKA values between the models in that cluster.

fig, axs = plt.subplots(n_clusters, 1, figsize=(8, 6 * n_clusters))

# Iterate through each cluster
for cluster in range(n_clusters):
    # Get the indices of the models that belong to this cluster
    indices = np.where(cluster_labels == cluster)[0]

    # Create a sub-grid containing only the CKA values between the models in this cluster
    sub_grid = grid[np.ix_(indices, indices)]

    # Draw the heatmap for this cluster
    axs[cluster].imshow(sub_grid, vmin=0, vmax=1, cmap="YlGnBu")
    axs[cluster].set_title(f"CKA between models in cluster {cluster + 1}")
    axs[cluster].set_xlabel("Model")
    axs[cluster].set_ylabel("Model")

    for (i, j), z in np.ndenumerate(sub_grid):
        axs[cluster].text(j, i, f"{z:.2f}", ha="center", va="center", color="w")

    # Get the model names for this cluster
    tick_labels_cluster = [tick_labels[i] for i in indices]

    # Add tick labels
    axs[cluster].set_xticks(np.arange(len(indices)))
    axs[cluster].set_xticklabels(tick_labels_cluster, rotation=90)
    axs[cluster].set_yticks(np.arange(len(indices)))
    axs[cluster].set_yticklabels(tick_labels_cluster, rotation=0)

plt.tight_layout()
plt.show()
# %%
from scipy.cluster.hierarchy import dendrogram, linkage

linkage_matrix = linkage(distance_matrix, method="complete")

# Create a dendrogram
dendrogram(linkage_matrix, labels=tick_labels, leaf_rotation=90)
plt.title("Hierarchical Clustering of Models")
plt.xlabel("Model")
plt.ylabel("Distance")
plt.show()

# %% [markdown]
# # Projection
# Let's visualize the distances between the models in 2D.

from sklearn.manifold import TSNE

# Apply t-SNE to the CKA values
tsne = TSNE(n_components=2, perplexity=3, n_iter=300)
tsne_results = tsne.fit_transform(1 - distance_matrix)

# Plot the t-SNE results
plt.figure(figsize=(10, 6))
plt.scatter(tsne_results[:, 0], tsne_results[:, 1])

# Add labels
for i, label in enumerate(tick_labels):
    plt.annotate(label, (tsne_results[i, 0], tsne_results[i, 1]))

plt.title("2D Graph of Model Clusters based on CKA")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.show()


# %%
