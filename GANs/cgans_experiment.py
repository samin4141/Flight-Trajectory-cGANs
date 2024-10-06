import torch
import torch.nn as nn
import math
import random
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import pandas as pd
import os
from funcs import delta_cyli_maker, delta_maker, draw_curved_line, draw_line, draw_spiral_line, rev_delta_cart_maker, rev_delta_cyli_maker
from trial_generator_discriminator import Generator, Discriminator

#Number of each to generate
class1 = 500 #Q3 to Q1
class2 = 500 #Q2 to Q4
length = 151
dim = 2

r1 = [50+random.random()*300 for i in range(0, class1)]
r2 = [50+random.random()*300 for i in range(0, class2)]

theta1 = [random.random()*3.14/2 for i in range(0, class1)]
theta2 = [3.14/2 + random.random()*3.14/2 for i in range(0, class2)]

lines1 = [draw_line(r1[i], theta1[i], length) for i in range(0, class1)]
lines2 = [draw_line(r2[i], theta2[i], length) for i in range(0, class2)]

labels = [0 for i in range(0, class1)] + [1 for i in range(0, class2)]

#Number of each to generate
class1 = 5000 #Q3 to Q1
class2 = 5000 #Q2 to Q4
length = 151
rot_angle = 12*3.14
dim = 2


theta1 = [rot_angle*i/length for i in range(0, length)]
theta2 = [rot_angle - rot_angle*i/length for i in range(0, length)]


lines1 = [draw_curved_line(1, theta1, length) for i in range(0, class1)]
lines2 = [draw_curved_line(2, theta2, length) for i in range(0, class2)]
labels = [0 for i in range(0, class1)] + [1 for i in range(0, class2)]


#Number of each to generate
class1 = 5000 #Q3 to Q1
class2 = 5000 #Q2 to Q4
length = 101
rot_angle = 4*3.14
dim = 2


theta1 = [rot_angle*i/length for i in range(0, length)]
theta2 = [rot_angle - rot_angle*i/length for i in range(0, length)]


lines1 = [draw_spiral_line(1, theta1, length) for i in range(0, class1)]
lines2 = [draw_spiral_line(2, theta2, length) for i in range(0, class2)]
labels = [0 for i in range(0, class1)] + [1 for i in range(0, class2)]

for i in range(0, 5):
    plt.plot(lines1[i][0], lines1[i][1])
plt.show()

for i in range(0, 5):
    plt.plot(lines2[i][0], lines2[i][1])

plt.show()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

#lines1_mod = delta_cyli_maker(lines1)
#lines2_mod = delta_cyli_maker(lines2)
#replot1 = rev_delta_cyli_maker(lines1_mod)
#replot2 = rev_delta_cyli_maker(lines2_mod)

lines1_mod = delta_maker(lines1)
lines2_mod = delta_maker(lines2)
replot1 = rev_delta_cart_maker(lines1_mod)
replot2 = rev_delta_cart_maker(lines2_mod)

length = length - 1

for i in range(0, 5):
  plt.plot(replot1[i][0], replot1[i][1])
plt.show()

for i in range(0, 5):
  plt.plot(replot2[i][0], replot2[i][1])
plt.show()

lines1_arr = torch.tensor(lines1_mod)
lines2_arr = torch.tensor(lines2_mod)
features_vec = torch.cat((lines1_arr,lines2_arr),0)

labels_vec = torch.tensor(labels)

input_dim = 4
n_classes = 2
out_shape = (length,dim)
output_dim = length*dim
bsize = 100
test_size = 12

cuda = True if torch.cuda.is_available() else False

        
dataset = TensorDataset(features_vec,labels_vec)
loader = DataLoader(dataset,batch_size=bsize)

adversarial_loss = torch.nn.BCELoss()
generator = Generator(n_classes, dim, length)
discriminator = Discriminator(n_classes, dim, length, device)

if str(device) == "cuda":
    print("GPU Enabled")
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5,0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5,0.999))

def plot_figures(generated_data, noise_labels):
    plt.figure(1)
    plt.clf()
    plt.figure(2)
    plt.clf()
    
    for i, raw_data in enumerate(generated_data):
        data = raw_data.detach().tolist()
        x = data[0]
        y = data[1]
        fixed_track = rev_delta_cart_maker([[x, y]])
        if noise_labels[i].item() == 0:
            plt.figure(1)
            plt.plot(fixed_track[0][0],fixed_track[0][1])
        else:
            plt.figure(2)
            plt.plot(fixed_track[0][0],fixed_track[0][1])
    plt.figure(1)
    plt.show()
    plt.figure(2)
    plt.show()

n_epochs = 2500

FloatTensor = torch.cuda.FloatTensor if device == "cuda" else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if device == "cuda" else torch.LongTensor

for epoch_idx in range(n_epochs):
    G_loss = []
    D_loss = []
    for batch_idx, data_input in enumerate(loader):
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
        if d_loss.data.item() > 0.15:
            optimizer_D.step()
        #optimizer_D.step()
        
        # Clear optimizer gradients

        G_loss.append(g_loss.data.item())
        # Evaluate the model
    if (epoch_idx)%50 == 49:

        with torch.no_grad():
            noise = torch.randn((test_size, 3*dim, 3*dim)).to(device) #Do only ten examples
            noise_labels = torch.randint(0, 2, (test_size,)).to(device)
            generated_data = generator(noise, noise_labels)
            
            plot_figures(generated_data, noise_labels)

    if (epoch_idx)%5 == 0:
        print('[%d/%d]: loss_d: %.3f, loss_g: %.3f' % (
            (epoch_idx), n_epochs, torch.mean(torch.FloatTensor(d_loss.to("cpu"))), torch.mean(torch.FloatTensor(g_loss.to("cpu")))))
        
        
