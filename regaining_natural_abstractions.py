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
# # Model A

model_a1 = train_model_A(a1_trainloader, testloader)

# %% [markdown]
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

model_b1 = train_unsimilar_model(model_a1, config, a1_trainloader, track_CKA=True)


# %% [markdown]
# # Regaining natural abstractions
# If we remove the CKA loss, we should see that the model regains the natural abstractions


def train_model_A_while_comparing(
    trainloader, testloader, base_model, epochs=None, model=None, variant_name="A"
):
    """
    Trains a model with the architecture of Model A
    :param trainloader: the training data
    :param testloader: the test data
    :param epochs: the number of epochs to train for, defaults to the number in config.json

    :return: the trained model
    """
    # open config.json and read it
    with open("config.json", "r") as f:
        config = json.load(f)
    if epochs is not None:
        config["epochs"] = epochs

    print(config)
    run_id = "Model_" + variant_name + "_" + datetime.now().strftime("%Y%m%d-%H%M%S")

    if model is None:
        model = ConvAutoencoder(config)

    wandb.init(project="CKA-different-representations", config=config, id=run_id)
    wandb.watch(model, log="all")
    criterion = torch.nn.MSELoss()
    # Hard coding device is dodgy but will work for now
    device = torch.device("cuda")

    hilbert_vectors_model_1 = base_model.get_full_Hilbert_rep(trainloader)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=config["optimizer"]["lr"])

    # Send model to device
    model.to(device)

    for epoch in range(config["epochs"]):
        # Training
        model.train()
        total_train_loss = 0

        for batch in trainloader:
            images, _ = batch
            images = images.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, images)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(trainloader)

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

    # save the model parameters to wandb (and not locally)
    torch.save(model.state_dict(), wandb.run.dir + "/model.pt")
    wandb.finish()

    return model


# %%

train_model_A_while_comparing(
    trainloader, testloader, model_a1, model=model_b1, variant_name="B_with_no_CKA"
)

# %%
