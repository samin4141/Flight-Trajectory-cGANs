
import torch.nn as nn
from funcs import delta_maker_torch
from torch.utils.data import DataLoader, Dataset


class Generator(nn.Module):
    def __init__(self, dim, length):
        super(Generator, self).__init__()
        
        self.dim = dim
        self.length = length
        
        self.noise1 = nn.Sequential(
            nn.Linear((3*dim)**2, 400),
            nn.BatchNorm1d(400),
            nn.ReLU()
        )

        self.noise2 = nn.Sequential(
            nn.Linear(400, 1500),
            nn.BatchNorm1d(1500),
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
            nn.Conv1d(dim*5, dim*5, 5, stride = 1, padding = 2),
            nn.Linear(200, 200),
        )

        self.hidden_layer6 = nn.Sequential(
            nn.Conv1d(dim*5, dim*1, 5, stride = 1, padding = 2),
            nn.Linear(200, length),
        )
        

    def forward(self, noise):
        batch_size = len(noise)
        noise = noise.flatten(start_dim=1)

        n_output = self.noise1(noise)
        n_output = self.noise2(n_output)

        inp = n_output.reshape(batch_size, self.dim, -1)
        output = self.hidden_layer3(inp)
        output = self.hidden_layer4(output)
        output = self.hidden_layer5(output)
        output = self.hidden_layer6(output)

        return output

class Discriminator(nn.Module):
    def __init__(self, dim, length, device):
        super(Discriminator, self).__init__()

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
            nn.Linear(50*dim, 30),
            nn.LeakyReLU(0.05),
            #nn.Dropout(0.2)
        )
        
        self.hidden_layer5 = nn.Sequential(
            nn.Linear(30, 30),
            nn.LeakyReLU(0.05),
            #nn.Dropout(0.2)
        )
        
        self.hidden_layer6 = nn.Sequential(
            nn.Linear(30, 1),
            nn.Sigmoid()
        )
        

    def forward(self, inp):
        output = self.layer1(inp)
        output = self.hidden_layer2(output)
        output = self.hidden_layer3(output)
        
        #Flatten layer occurs here
        output = output.flatten(start_dim=1)
        output = output[:, None,:]
        output = self.hidden_layer4(output)
        output = self.hidden_layer5(output)
        output = self.hidden_layer6(output)
        
        output = output.squeeze(-1)
        return output.to(self.device)
    
class TrajectoryDataset(Dataset):
    def __init__(self, data_arr, label_arr, transform=None, target_transform=None):
        self.data_arr = data_arr
        self.label_arr = label_arr
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.label_arr)

    def __getitem__(self, idx):
        trajectory = self.data_arr[idx]
        label = self.label_arr[idx]
        
        trajectory = delta_maker_torch(trajectory)

        if self.target_transform:
            label = self.target_transform(label)
        return trajectory, label