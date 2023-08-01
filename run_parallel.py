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
import os


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

# Vars
MODEL_NUMBER = os.environ["MODEL_NUMBER"]
MODEL_NUMBER = int(MODEL_NUMBER)


# %%
model_a1 = ConvAutoencoder(config)
model_a1.load_state_dict(torch.load("model_a1.pt"))
model_a1.to(device)


print("Loaded!")

# CUDA_VISIBLE_DEVICES=0 MODEL_NUMBER=5 python run_parallel.py &
# CUDA_VISIBLE_DEVICES=1 MODEL_NUMBER=6 python run_parallel.py &
# CUDA_VISIBLE_DEVICES=2 MODEL_NUMBER=7 python run_parallel.py &
# CUDA_VISIBLE_DEVICES=3 MODEL_NUMBER=8 python run_parallel.py
# #


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

    run_id = f"ModelB_{MODEL_NUMBER}__" + datetime.now().strftime("%Y%m%d-%H%M%S")

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

model_b = train_unsimilar_model(model_a1, config, trainloader, track_CKA=False)


# %%

torch.save(model_b.state_dict(), f"model_b{MODEL_NUMBER}.pt")
