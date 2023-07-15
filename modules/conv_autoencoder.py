import torch.nn as nn
import torch

class ConvAutoencoder(nn.Module):
    def __init__(self, config: dict):
        super(ConvAutoencoder, self).__init__()

        # Define dimensions for better readability
        conv_dim = config["arch"]["conv_dim"] 
        self.latent_dim = config["arch"]["latent_dim"]

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, conv_dim, kernel_size=3, stride=2, padding=1),  # (B, 16, 14, 14)
            nn.ReLU(),
            nn.Conv2d(conv_dim, 2*conv_dim, kernel_size=3, stride=2, padding=1),  # (B, 32, 7, 7)
            nn.ReLU(),
            nn.Flatten(),  # (B, 32*7*7)
            nn.Linear((2*conv_dim)*7*7, self.latent_dim))  # (B, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, (2*conv_dim)*7*7),  # (B, 32*7*7)
            nn.ReLU(),
            nn.Unflatten(1, (2*conv_dim, 7, 7)),  # (B, 32, 7, 7)
            nn.ConvTranspose2d(2*conv_dim, conv_dim, kernel_size=3, stride=2, padding=1, output_padding=1),  # (B, 16, 14, 14)
            nn.ReLU(),
            nn.ConvTranspose2d(conv_dim, 1, kernel_size=3, stride=2, padding=1, output_padding=1))  # (B, 1, 28, 28)

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def get_Hilbert_rep(self, dataloader):
        # for each batch, get the encoded representation and concatenate into one tensor of shape (N, latent_dim) 
        # where N is the total number of images in the dataset
        Hilbert_rep = torch.empty(0, self.latent_dim)
        for batch_idx, (data, target) in enumerate(dataloader):
            Hilbert_rep = torch.cat((Hilbert_rep, self.encoder(data)), 0)
        return Hilbert_rep
    
        