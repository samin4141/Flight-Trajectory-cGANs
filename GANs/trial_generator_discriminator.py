import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import random
import torch


class Generator(nn.Module):
    def __init__(self, n_classes, dim, length):
        super(Generator, self).__init__()
        self.n_classes = n_classes
        self.dim = dim
        self.length = length
        
        self.noise1 = nn.Sequential(
            nn.Linear((3*dim)**2, 400),
            nn.BatchNorm1d(400),
            nn.ReLU()
        )

        self.noise2 = nn.Sequential(
            nn.Linear(400, 1300),
            nn.BatchNorm1d(1300),
            nn.ReLU(),
        )

        self.label1 = nn.Sequential(
            nn.Linear(n_classes, 10),
            nn.BatchNorm1d(10),
            nn.ReLU()
        )

        self.label2 = nn.Sequential(
            nn.Linear(10, 200),
            nn.BatchNorm1d(200),
            nn.ReLU(),
        )
        
        self.hidden_layer3 = nn.Sequential(
            nn.Conv1d(dim, dim*10, 5, stride = 1, padding = 2),
            nn.Linear(int(1500/dim), 200),
            nn.Upsample(scale_factor = 2, mode='linear'),
            nn.BatchNorm1d(dim*10, affine=False),
            nn.ReLU()
        )
        
        self.hidden_layer4 = nn.Sequential(
            nn.Conv1d(dim*10, dim*5, 5, stride = 1, padding = 2),
            nn.Linear(400, 100),
            nn.Upsample(scale_factor = 2, mode='linear'),
            nn.BatchNorm1d(dim*5, affine=False),
            nn.ReLU()
        )

        self.hidden_layer5 = nn.Sequential(
            nn.Conv1d(dim*5, dim*1, 5, stride = 1, padding = 2),
            nn.Linear(200, length),
        )

    def forward(self, noise, labels):        
        c = nn.functional.one_hot(labels, num_classes = self.n_classes).to(torch.float32)
        noise = noise.flatten(start_dim=1)
        
        n_output = self.noise1(noise)
        n_output = self.noise2(n_output)
        
        l_output = self.label1(c)
        l_output = self.label2(l_output)
        
        inp = torch.cat([n_output, l_output], 1)
        output = inp.reshape(len(labels), self.dim, -1)
        
        output = self.hidden_layer3(output)
        output = self.hidden_layer4(output)
        output = self.hidden_layer5(output)

        return output
      
    
class Discriminator(nn.Module):
    def __init__(self, n_classes, dim, length, device):
        super(Discriminator, self).__init__()
        
        self.n_classes = n_classes
        self.dim = dim
        self.length = length
        self.device = device

        self.layer1 = nn.Sequential(
            nn.Linear(length, 200),
            nn.Conv1d(dim, dim*20, 5, stride = 1, padding = 2),
            nn.LeakyReLU(0.05),
            nn.Dropout(0.05)
        )
        self.hidden_layer2 = nn.Sequential(
            nn.Conv1d(dim*20, dim*5, 5, stride = 1, padding = 2),
            nn.Linear(200, 150),
            nn.LeakyReLU(0.05),
            nn.Dropout(0.05)
        )

        self.hidden_layer3 = nn.Sequential(
            nn.Conv1d(dim*5, dim, 5, stride = 1, padding = 2),
            nn.Linear(150, 50),
            nn.LeakyReLU(0.05),
            nn.Dropout(0.05)
        )

        self.hidden_layer4 = nn.Sequential(
            nn.Linear(50*2 + n_classes, 30),
            nn.LeakyReLU(0.05),
            #nn.Dropout(0.2)
        )
        
        self.hidden_layer5 = nn.Sequential(
            nn.Linear(30, 1),
            nn.Sigmoid()
        )
        

    def forward(self, inp, labels):
        c = nn.functional.one_hot(labels, num_classes = self.n_classes)
        inp = inp.reshape(-1, self.dim, self.length)
        output = self.layer1(inp)
        output = self.hidden_layer2(output)
        output = self.hidden_layer3(output)
        
        #Flatten layer occurs here
        output = torch.cat([output.flatten(start_dim=1), c], 1)
        output = output[:, None,:]
        output = self.hidden_layer4(output)
        output = self.hidden_layer5(output)
        
        output = output.squeeze(-1)
        return output.to(self.device)
