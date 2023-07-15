import torch.optim as optim
import torch
import torch.nn as nn
from metrics import CKA_function
import tqdm
import wandb

class Trainer:
    def __init__(self, model, data_loader, config, natural_Hilbert_vectors = None):
        self.model = model
        self.data_loader = data_loader
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.epochs = config["trainer"]["epochs"]
        self.save_dir = config["trainer"]["save_dir"]
        self.save_period = config["trainer"]["save_period"]

        self.default_Hilbert_vectors = natural_Hilbert_vectors

        optimizer_type = config["optimizer"]["type"]
        optimizer_args = config["optimizer"]["args"]
        
        self.optimizer = getattr(optim, optimizer_type)(model.parameters(), **optimizer_args)
        
        self.loss_fn = nn.MSELoss()

    def train(self):
        for epoch in range(self.epochs):
            self.model.train()  # set the model to training mode

            for batch_idx, (data, target) in tqdm(self.data_loader):
                data = data.to(device=self.device)
                target = target.to(device=self.device)

                self.optimizer.zero_grad()  # reset gradients

                output = self.model(data)  # forward pass

                loss = self.loss_fn(output, target)  # compute loss

                loss.backward()  # backward pass 

                wandb.log({"loss": loss.item()})

                self.optimizer.step()  # update weights
            
            # save model every save_period epochs
            if (epoch + 1) % self.save_period == 0:
                torch.save(self.model.state_dict(), f"{self.save_dir}/model_{epoch+1}.pth")
                
            print(f"Epoch: {epoch+1}/{self.epochs}, Loss: {loss.item()}")