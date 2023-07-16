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

plt.show()  # Display all 5 images


# %%
import torch.optim as optim

model_a1 = train_model_A(a1_trainloader, testloader)

# %%

import time

model = ConvAutoencoder(config)

# # Let's compare get_latent_hilbert_rep vs get_full_Hilbert_rep

# for batch in a1_trainloader:
#     images, _ = batch
#     images = images.to(device)

#     start_time = time.time()
#     rep_1 = model.get_latent_Hilbert_rep_batch(images)
#     rep_2 = model.get_latent_Hilbert_rep_batch(images)
#     # size:
#     print(rep_1.shape)
#     cka = CKA_function(rep_1, rep_2)
#     end_time = time.time()
#     print("get_latent_Hilbert_rep_batch: ", end_time - start_time)

#     start_time = time.time()
#     rep_1 = model.get_full_Hilbert_rep_batch(images)
#     rep_2 = model.get_full_Hilbert_rep_batch(images)
#     # size:
#     print(rep_1.shape)
#     cka = CKA_function(rep_1, rep_2)
#     end_time = time.time()
#     print("get_full_Hilbert_rep_batch: ", end_time - start_time)
#     break

# Results:
# get_latent_Hilbert_rep_batch:  0.0013217926025390625
# get_full_Hilbert_rep_batch:  0.00799107551574707


# %%

display_encoded_samples(model_a1, testset)

# %%

# import CKA

import torch.onnx


def train_unsimilar_model(base_model, epsilon, epochs):
    model = ConvAutoencoder(config)
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.to(device)

    hilbert_vectors_model_1 = base_model.get_full_Hilbert_rep(trainloader)

    run_id = "ModelB_" + datetime.now().strftime("%Y%m%d-%H%M%S")

    wandb.init(project="CKA-different-representations", config=config, id=run_id)
    wandb.watch(model, log="all")

    # mark this as a model B

    for _ in range(epochs):
        # Training
        model.train()
        total_train_loss = 0

        for batch in a2_trainloader:
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

            loss = criterion(outputs, images) + epsilon * CKA

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(a2_trainloader)

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
    wandb.finish()

    return model


# %%

epsilon = 0.05
epochs = 15
model_b1 = train_unsimilar_model(model_a1, epsilon, epochs)

# %%

model_b2 = train_unsimilar_model(model_a1, epsilon, epochs)

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
