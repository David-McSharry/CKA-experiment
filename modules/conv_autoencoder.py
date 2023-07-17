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
        N = len(dataloader.dataset)
        Hilbert_rep = torch.empty(N, self.latent_dim).to(self.device)

        start = 0
        for batch_idx, (data, target) in enumerate(dataloader):
            latent_rep_batch = self.get_latent_Hilbert_rep_batch(data)
            end = start + latent_rep_batch.size(0)
            Hilbert_rep[start:end] = latent_rep_batch
            start = end

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
        N = len(dataloader.dataset)
        D = self.get_full_Hilbert_rep_batch(next(iter(dataloader))[0]).size(1)
        full_Hilbert_rep = torch.empty(N, D).to(self.device)

        start = 0
        for batch_idx, (data, target) in enumerate(dataloader):
            hilbert_batch = self.get_full_Hilbert_rep_batch(data)
            end = start + hilbert_batch.size(0)
            full_Hilbert_rep[start:end] = hilbert_batch
            start = end

        return full_Hilbert_rep
