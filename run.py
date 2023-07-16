# %% ---------------------------------------

import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import lovely_tensors as lt
import wandb
import json
from datetime import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




# %% ---------------------------------------


transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])

# Download and load the full training images
full_trainset = datasets.MNIST('./data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(full_trainset, batch_size=64)

a1_trainset, a2_trainset = torch.utils.data.random_split(full_trainset, [len(full_trainset)//2, len(full_trainset)//2])

a1_trainloader = torch.utils.data.DataLoader(a1_trainset, batch_size=64)
a2_trainloader = torch.utils.data.DataLoader(a2_trainset, batch_size=64)

# Download and load the full test images
full_testset = datasets.MNIST('./data/', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(full_testset, batch_size=64)





# %% ---------------------------------------


lt.monkey_patch()

image, label = next(iter(trainloader))
plt.imshow(image[0].numpy().squeeze(), cmap='gray_r')



# %% ---------------------------------------



# load in config.json
with open("config.json", 'r') as f:
    config = json.load(f)


print(json.dumps(config, indent=4, sort_keys=True))


# %% ---------------------------------------


from modules.conv_autoencoder import ConvAutoencoder


model = ConvAutoencoder(config)
# do a forward pass on a single image
output = model(image)

print(output.shape)


# %% ---------------------------------------

# import time
# hilbert_trainloader = torch.utils.data.DataLoader(full_trainset, batch_size=100000)

# start_time = time.time()

# hilbert_vectors = model.get_Hilbert_rep(

# end_time = time.time()

# print("Execution time:", end_time - start_time, "seconds")

# print(hilbert_vectors.shape)


# %% ---------------------------------------


import torch.optim as optim

epochs = 5
# Loss function
criterion = torch.nn.MSELoss()

# Optimizer
optimizer = optim.Adam(model.parameters(), lr = 0.001)



# Send model to device
model.to(device)

for epoch in range(epochs):

    # Training
    model.train()
    total_train_loss = 0

    for batch in a1_trainloader:
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

    avg_train_loss = total_train_loss / len(a1_trainloader)

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
    # send it to the device
    image = image.to(device)
    # send it to the model
    output = model(image)
    # plot the original and reconstructed images
    plt.imshow(image[0].cpu().numpy().squeeze(), cmap='gray_r')
    plt.show()
    plt.imshow(output[0].cpu().numpy().squeeze(), cmap='gray_r')
    plt.show()

hilbert_vectors_model_1 = model.get_Hilbert_rep(trainloader)



# %%

# import CKA
from metrics.CKA import CKA_function
model2 = ConvAutoencoder(config)
import torch.onnx
import time
# train another model with the same config and the same loop as above

criterion = torch.nn.MSELoss()
optimizer = optim.Adam(model2.parameters(), lr = 0.001)
model2.to(device)

epochs = 15

config["run_id"] = datetime.now().strftime("%Y%m%d-%H%M%S")
print(json.dumps(config, indent=4))

wandb.init(project="CKA-different-representations", config=config, id=config["run_id"])


CKA_arr = []

epsilon =0.05

for epoch in range(epochs):

        # Training
        model2.train()
        total_train_loss = 0

        for batch in a2_trainloader:
            images, _ = batch
            images = images.to(device)

            # Forward pass
            outputs = model2(images)

            batch_hilbert_vectors_model_1 = model.get_Hilbert_rep_batch(images)
            batch_hilbert_vectors_model_2 = model2.get_Hilbert_rep_batch(images)
            CKA = CKA_function(batch_hilbert_vectors_model_1, batch_hilbert_vectors_model_2)

            loss = criterion(outputs, images) + epsilon *  CKA

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(a2_trainloader)

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

# kill wandb process
wandb.finish()

# %%

#plot the CKA values
CKA_arr = [x.cpu().numpy() for x in CKA_arr]
plt.plot(CKA_arr)
plt.show()


# %%

# test the model
model2.eval()
with torch.no_grad():
    # get a random test image
    iterator = iter(testloader)
    next(iterator)
    next(iterator)
    image, label = next(iterator)
    # send it to the device
    image = image.to(device)
    # send it to the model
    output = model2(image)
    # plot the original and reconstructed images
    plt.imshow(image[0].cpu().numpy().squeeze(), cmap='gray_r')
    plt.show()
    plt.imshow(output[0].cpu().numpy().squeeze(), cmap='gray_r')
    plt.show()

# %%
# mke tenaor of size latent_dim

vec = torch.tensor([[1.0, 3.0, 3.0, 4.0, 4.0, 8.0, 3.0, 4.0, 1.0, 3.3, 1.0, 2.0]]).to(device)

with torch.no_grad():
    output = model.decoder(vec)

# visualize the output
plt.imshow(output.cpu().numpy().squeeze(), cmap='gray_r')
# %%
