# %% ---------------------------------------

import torch 
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import lovely_tensors as lt
import wandb
import json
from datetime import datetime




# %% ---------------------------------------

transform = transforms.Compose([transforms.ToTensor(), 
                                transforms.Normalize((0.5,), (0.5,))])

n = 200

# Download and load the first 'n' training images
full_trainset = datasets.MNIST('./data/', download=True, train=True, transform=transform)
trainset = torch.utils.data.Subset(full_trainset, range(n))
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Download and load the first 'n' test images
full_testset = datasets.MNIST('./data/', download=True, train=False, transform=transform)
testset = torch.utils.data.Subset(full_testset, range(n))
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)





# %% ---------------------------------------


lt.monkey_patch()

image, label = next(iter(trainloader))
plt.imshow(image[0].numpy().squeeze(), cmap='gray_r')



# %% ---------------------------------------



# load in config.json
with open("config.json", 'r') as f:
    config = json.load(f)





# %% ---------------------------------------


from modules.conv_autoencoder import ConvAutoencoder


model = ConvAutoencoder(config)
# do a forward pass on a single image
output = model(image)

print(output.shape)


# %% ---------------------------------------

from modules.conv_autoencoder import ConvAutoencoder

model = ConvAutoencoder(config)
hilbert_vectors = model.get_Hilbert_rep(trainloader)

print(hilbert_vectors.shape)


# %% ---------------------------------------


import torch.optim as optim

# Loss function
criterion = torch.nn.MSELoss()

# Optimizer
optimizer = optim.Adam(model.parameters(), lr = 0.001)

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


# Send model to device
model.to(device)

epochs = 50

for epoch in range(epochs):

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


    avg_val_loss = total_val_loss / len(testloader)

    print(f"Epoch: {epoch + 1}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")






# %%


# test the model 
model.eval()
with torch.no_grad():
    # get a random test image
    image, label = next(iter(testloader))
    # send it to the model
    output = model(image)
    # plot the original and reconstructed images
    plt.imshow(image[0].numpy().squeeze(), cmap='gray_r')
    plt.show()
    plt.imshow(output[0].numpy().squeeze(), cmap='gray_r')
    plt.show()

hilbert_vectors_model_1 = model.get_Hilbert_rep(trainloader)



# %%

# import CKA
from metrics.CKA import CKA_function
model2 = ConvAutoencoder(config)
# train another model with the same config and the same loop as above

criterion = torch.nn.MSELoss()
optimizer = optim.Adam(model2.parameters(), lr = 0.001)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model2.to(device)

epochs = 500

# update config with run_id
config["run_id"] = datetime.now().strftime('%Y%m%d%H%M%S')
print(json.dumps(config, indent=4))

wandb.init(project="CKA-different-representations", config=config, id=config["run_id"])


CKA_arr = []

for epoch in range(epochs):
    
        # Training
        model2.train()
        total_train_loss = 0
    
        for batch in trainloader:
            images, _ = batch
            images = images.to(device)
    
            # Forward pass
            outputs = model2(images)
            loss = criterion(outputs, images)
    
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            total_train_loss += loss.item()
    
        avg_train_loss = total_train_loss / len(trainloader)
    
        # Validation
        model2.eval()
        total_val_loss = 0
    
        with torch.no_grad():
            for batch in testloader:
                images, _ = batch
                images = images.to(device)
    
                outputs = model2(images)
                loss = criterion(outputs, images)
    
                total_val_loss += loss.item()

            hilbert_vectors_model_2 = model2.get_Hilbert_rep(trainloader)
            print(hilbert_vectors_model_1)
            print(hilbert_vectors_model_2)
            CKA = CKA_function(hilbert_vectors_model_1, hilbert_vectors_model_2)
            print(CKA)
            CKA_arr.append(CKA)

            wandb.log({"CKA": CKA})
            wandb.log({"Training Loss": avg_train_loss})
            wandb.log({"Validation Loss": avg_val_loss})


    
        avg_val_loss = total_val_loss / len(testloader)

        # get hilbert vectors of the training set

        print(f"Epoch: {epoch + 1}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")


# %% 

#plot the CKA values
plt.plot(CKA_arr)
plt.show()

 # %%




# %%


