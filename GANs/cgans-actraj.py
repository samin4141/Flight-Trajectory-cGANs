import torch
import torch.nn as nn
import math
import random
import numpy as np
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import pandas as pd
import os
from funcs import plot_figures, rev_delta_maker
from trial_generator_discriminator import Generator, Discriminator, TrajectoryDataset


bsize = 1000
test_size = 12
dataloader = DataLoader(torch.load('/kaggle/input/dataset/dataset.pt'), batch_size=bsize, shuffle=True, drop_last=True, num_workers=5)

sample_data = next(iter(dataloader))
plt.figure(1)
ax = plt.axes(projection='3d')

for i in range(10):
    line = rev_delta_maker(sample_data[0][i])
    X, Y, Z = line[0], line[1], line[2]
    ax.plot3D(X, Y, Z)  # Plot contour curves

plt.show()



length = len(sample_data[0][0][0])
dim = len(sample_data[0][0][:,0])

print(length)
print(dim)

n_classes = 5 #Fix manually...
out_shape = (length,dim)
output_dim = length*dim

cuda = True if torch.cuda.is_available() else False


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

adversarial_loss = torch.nn.MSELoss()
generator = Generator(n_classes, dim, length)
discriminator = Discriminator(n_classes, dim, length, device)

gen_trainable_params = sum(
    p.numel() for p in generator.parameters() if p.requires_grad
)

disc_trainable_params = sum(
    p.numel() for p in discriminator.parameters() if p.requires_grad
)
gen_txt = "Generator Parameters: {g}"
print(gen_txt.format(g = gen_trainable_params))
dis_txt = "Discrminator Parameters: {g}"
print(dis_txt.format(g = disc_trainable_params))


if str(device) == "cuda":
    print("GPU Enabled")
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5,0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5,0.999))

n_epochs = 150

FloatTensor = torch.cuda.FloatTensor if device == "cuda" else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if device == "cuda" else torch.LongTensor

for epoch_idx in range(n_epochs):
    G_loss = []
    D_loss = []
    for batch_idx, data_input in enumerate(dataloader):
        #Adversarial ground truths
        valid = torch.autograd.Variable(FloatTensor(bsize, 1).fill_(0.9), requires_grad=False).to(device)
        fake = torch.autograd.Variable(FloatTensor(bsize, 1).fill_(0.0), requires_grad=False).to(device)
        
        #Real Input
        real_tracks = data_input[0]
        real_labels = data_input[1]
                
        ### Training Generator ###
        optimizer_G.zero_grad()

        noise = torch.randn((bsize, 3*dim, 3*dim)).to(device)
        noise_labels = torch.randint(0, 2, (bsize,)).to(device)
        generated_data = generator(noise, noise_labels)
        
        #Update the generator loss function based on its ability to trick
        validity = discriminator(generated_data, noise_labels)
        g_loss = adversarial_loss(validity, valid)
        
        g_loss.backward()
        optimizer_G.step()
        
        ### Training Discriminator ###
        
        mreal_track = real_tracks.to(device) 
        mreal_labels = real_labels.to(device)
        
        # Clear optimizer gradients        
        optimizer_D.zero_grad()
        # Forward pass with true data as input
        validity_real = discriminator(mreal_track,mreal_labels)
        d_real_loss = adversarial_loss(validity_real, valid)
        validity_fake = discriminator(generated_data.detach(),noise_labels)
        d_fake_loss = adversarial_loss(validity_fake, fake)

        # Average the loss
        d_loss = (
            d_real_loss + d_fake_loss
        ) / 2
        
        d_loss.backward()
        #print(d_loss)
        if d_loss.data.item()*5 > g_loss.data.item():
            optimizer_D.step()
        #optimizer_D.step()
        
        # Clear optimizer gradients

        G_loss.append(g_loss.data.item())

    if (epoch_idx)%5 == 0:
        print('[%d/%d]: loss_d: %.3f, loss_g: %.3f' % (
            (epoch_idx), n_epochs, torch.mean(torch.FloatTensor(d_loss.to("cpu"))), torch.mean(torch.FloatTensor(g_loss.to("cpu")))))

torch.save(generator.state_dict(), "generator.h5")
torch.save(discriminator.state_dict(), "discriminator.h5")