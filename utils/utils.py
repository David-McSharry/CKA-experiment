from argparse import ArgumentParser
import json
from datetime import datetime


def get_flags():
    args = ArgumentParser(description="Tests on natural abstractions with CKA")
    args.add_argument(
        "--run_id",
        default=None,
        type=str,
        help="Unique Identifier for training processes. Used to save checkpoints and training log. Timestamp is being used as default",
    )

    return args


def update_config(args: ArgumentParser):
    # Parse the command line arguments
    parsed_args = args.parse_args()

    # Load the existing config file
    with open("config.json", "r") as f:
        config = json.load(f)

    # Directly modify the config
    if parsed_args.run_id:
        config["run_id"] = parsed_args.run_id

    else:
        config["run_id"] = datetime.now().strftime("%Y%m%d%H%M%S")

    return config


import json
from datetime import datetime

import lovely_tensors as lt
import matplotlib.pyplot as plt
import torch
import wandb
from torchvision import datasets, transforms

device = torch.device("cuda")


def display_encoded_samples(model, dataset):
    model.eval()
    with torch.no_grad():
        for i in range(5):
            # get a random test image
            image, label = dataset[i]
            # send it to the device
            image = image.to(device)
            # encode the image
            encoded = model.encoder(image.unsqueeze(0))
            # decode the encoded image
            decoded = model.decoder(encoded)
            # plot the original and reconstructed images (hide the axes)
            plt.subplot(2, 5, i + 1)
            plt.imshow(image.cpu().numpy().squeeze(), cmap="gray_r")
            plt.subplot(2, 5, i + 6)
            plt.imshow(decoded.cpu().numpy().squeeze(), cmap="gray_r")
            plt.axis("off")
        plt.show()
