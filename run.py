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
import torch.optim as optim


device = torch.device("cuda:0")

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


# plot the first 5 images to make sure


# %%


images, labels = next(iter(trainloader))

fig, axs = plt.subplots(
    nrows=1, ncols=5, figsize=(10, 2)
)  # Create subplots for 5 images
for i in range(5):  # Repeat for each of the first 5 images
    axs[i].imshow(
        images[i].numpy().squeeze(), cmap="gray_r"
    )  # Plot each image on a separate subplot
    axs[i].axis("off")  # Remove axes for cleaner look


plt.show()  # Display all 5 imageA


# %%

model_a1 = train_model_A(a1_trainloader, testloader)


# %%

display_encoded_samples(model_a1, testset)

# %%md
# # Model B


def train_unsimilar_model(
    base_model, config, trainloader_unsimilar_model, track_CKA=False
):
    model = ConvAutoencoder(config)
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.to(device)

    hilbert_vectors_model_1 = base_model.get_full_Hilbert_rep(trainloader)

    run_id = "ModelB_" + datetime.now().strftime("%Y%m%d-%H%M%S")

    wandb.init(project="CKA-different-representations", config=config, id=run_id)
    wandb.watch(model, log="all")

    # mark this as a model B

    for _ in range(config["epochs"]):
        # Training
        model.train()

        total_train_loss = 0

        for batch in trainloader_unsimilar_model:
            images, _ = batch
            images = images.to(device)

            # Forward pass
            outputs = model(images)

            batch_hilbert_vectors_model_1 = base_model.get_full_Hilbert_rep_batch(
                images
            )
            batch_hilbert_vectors_model_2 = model.get_full_Hilbert_rep_batch(images)
            CKA = CKA_function(
                batch_hilbert_vectors_model_1, batch_hilbert_vectors_model_2
            )

            del batch_hilbert_vectors_model_1
            del batch_hilbert_vectors_model_2

            main_loss = criterion(outputs, images)
            cka_loss = config["CKA_loss_weight"] * CKA

            loss = main_loss + cka_loss

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += main_loss.item()

        avg_train_loss = total_train_loss / len(trainloader_unsimilar_model)

        # Validation
        model.eval()
        total_val_loss = 0

        with torch.no_grad():
            for batch in testloader:
                images, _ = batch
                images = images.to(device)

                outputs = model(images)
                loss = criterion(outputs, images)

                total_val_loss += loss.item()
            if track_CKA:
                print("Calculating CKA")
                hilbert_vectors_model_2 = model.get_full_Hilbert_rep(trainloader)
                CKA = CKA_function(hilbert_vectors_model_1, hilbert_vectors_model_2)
                wandb.log({"CKA": CKA})

            wandb.log(
                {
                    "train_loss": avg_train_loss,
                    "val_loss": total_val_loss / len(testloader),
                }
            )

    # kill wandb process
    torch.save(model.state_dict(), wandb.run.dir + "/model.pt")

    wandb.finish()

    return model


# %%

# load model_a1.pt
# model_a1 = ConvAutoencoder(config)
# model_a1.load_state_dict(torch.load("model_a1.pt"))
model_b5 = train_unsimilar_model(model_a1, config, trainloader, track_CKA=False)


# %%

# save model 2
torch.save(model_b5.state_dict(), "model_b5.pt")


# %%
# Are model_b1 and model_b2 similar?

hilbert_vectors_model_b1 = model_b1.get_latent_Hilbert_rep(trainloader)
hilbert_vectors_model_b2 = model_b2.get_latent_Hilbert_rep(trainloader)
print(CKA_function(hilbert_vectors_model_b1, hilbert_vectors_model_b2))


# %%
# test the model

display_encoded_samples(model_b1, testset)

# %%
# mke tenaor of size latent_dim

vec = torch.tensor([[1.0, 3.0, 3.0, 4.0, 4.0, 8.0, 3.0, 4.0, 1.0, 3.3, 1.0, 2.0]]).to(
    device
)

with torch.no_grad():
    output = model_a1.decoder(vec)

# visualize the output
plt.imshow(output.cpu().numpy().squeeze(), cmap="gray_r")
# %%

model_nums = [1, 2, 3, 4]

# for each pair of models, bet the CKA between them
import numpy as np

grid = np.zeros((4, 4))

with torch.no_grad():
    for model_num_1 in model_nums:
        for model_num_2 in model_nums:
            if model_num_1 != model_num_2:
                model_1 = ConvAutoencoder(config)
                model_1.load_state_dict(torch.load(f"model_b{model_num_1}.pt"))
                model_2 = ConvAutoencoder(config)
                model_2.load_state_dict(torch.load(f"model_b{model_num_2}.pt"))
                hilbert_vectors_model_1 = model_1.get_latent_Hilbert_rep(a1_trainloader)
                hilbert_vectors_model_2 = model_2.get_latent_Hilbert_rep(a1_trainloader)
                CKA = CKA_function(hilbert_vectors_model_1, hilbert_vectors_model_2)
                print(
                    f"CKA between model_b{model_num_1} and model_b{model_num_2}: {CKA}"
                )
                grid[model_num_1 - 1, model_num_2 - 1] = CKA
                del model_1
                del model_2
                del hilbert_vectors_model_1
                del hilbert_vectors_model_2


# %%

# make the diagonal values 1
np.fill_diagonal(grid, 1)

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))

# Create a heatmap
sns.heatmap(grid, vmin=0, vmax=1, annot=True, cmap="YlGnBu")

# Add labels
plt.xlabel("Models")
plt.ylabel("Models")

# Add title
plt.title("CKA between models")

# Show plot
plt.show()


# %%
