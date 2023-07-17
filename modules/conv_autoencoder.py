import torch.nn as nn
import torch


class ConvAutoencoder(nn.Module):
    def __init__(self, config: dict):
        super(ConvAutoencoder, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Define dimensions for readability
        conv_dim = config["arch"]["conv_dim"]
        self.latent_dim = config["arch"]["latent_dim"]

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(
                1, conv_dim, kernel_size=3, stride=2, padding=1
            ),  # (B, 16, 14, 14)
            nn.ReLU(),
            nn.Conv2d(
                conv_dim, 2 * conv_dim, kernel_size=3, stride=2, padding=1
            ),  # (B, 32, 7, 7)
            nn.ReLU(),
            nn.Flatten(),  # (B, 32*7*7)
            nn.Linear((2 * conv_dim) * 7 * 7, self.latent_dim),  # (B, latent_dim)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, (2 * conv_dim) * 7 * 7),  # (B, 32*7*7)
            nn.ReLU(),
            nn.Unflatten(1, (2 * conv_dim, 7, 7)),  # (B, 32, 7, 7)
            nn.ConvTranspose2d(
                2 * conv_dim,
                conv_dim,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),  # (B, 16, 14, 14)
            nn.ReLU(),
            nn.ConvTranspose2d(
                conv_dim, 1, kernel_size=3, stride=2, padding=1, output_padding=1
            ),  # (B, 1, 28, 28)
        )

        self.to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        return self.decoder(self.encoder(x))

    def get_latent_Hilbert_rep(self, dataloader):
        # for each batch, get the encoded representation and concatenate into one tensor of shape (N, latent_dim)
        # where N is the total number of images in the dataset
        Hilbert_rep = torch.empty(0, self.latent_dim).to(self.device)
        for batch_idx, (data, target) in enumerate(dataloader):
            Hilbert_rep = torch.cat(
                (Hilbert_rep, self.get_latent_Hilbert_rep_batch(data)), 0
            )
        return Hilbert_rep

    def get_latent_Hilbert_rep_batch(self, batch):
        return self.encoder(batch.to(self.device))

    def get_full_Hilbert_rep_batch(self, batch):
        activations = []
        batch = batch.to(self.device)
        for layer in self.encoder:
            batch = layer(batch)
            activations.append(batch.view(batch.size(0), -1))

        # Skip the last layer of the decoder (the output layer)
        for layer in self.decoder[:-1]:
            batch = layer(batch)
            activations.append(batch.view(batch.size(0), -1))
        return torch.cat(activations, dim=1)

    def get_full_Hilbert_rep(self, dataloader):
        # for each batch, get the full Hilbert representation and concatenate into one tensor of shape (N, ?)
        # where N is the total number of images in the dataset
        full_Hilbert_rep = torch.empty(
            0, self.get_full_Hilbert_rep_batch(next(iter(dataloader))[0]).size(1)
        ).to(self.device)
        for batch_idx, (data, target) in enumerate(dataloader):
            print(f"Batch {batch_idx}")
            full_Hilbert_rep = torch.cat(
                (full_Hilbert_rep, self.get_full_Hilbert_rep_batch(data)), 0
            )
        return full_Hilbert_rep
