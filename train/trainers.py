import torch
import torch.optim as optim
import wandb
from datetime import datetime
from modules.conv_autoencoder import ConvAutoencoder
import json

DEVICE = torch.device("cuda:0")


def train_model_A(trainloader, testloader, epochs=None, model=None, variant_name="A"):
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
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=config["optimizer"]["lr"])

    # Send model to device
    model.to(DEVICE)

    for epoch in range(config["epochs"]):
        # Training
        model.train()
        total_train_loss = 0

        for batch in trainloader:
            images, _ = batch
            images = images.to(DEVICE)

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
                images = images.to(DEVICE)

                outputs = model(images)
                loss = criterion(outputs, images)

                total_val_loss += loss.item()

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
