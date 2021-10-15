# Importing Required Packages
import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
import numpy as np
import torchvision
import torchvision.datasets as ds
import torchvision.transforms as tf
import matplotlib.pyplot as plt
from sklearn import *
params = {"text.color" : "k",
          "ytick.color" : "k",
          "xtick.color" : "k",
          "axes.labelcolor" : "k",
          "axes.edgecolor" : "k"}
plt.rcParams.update(params)
import math
from random import sample
from google.colab import files

# Generator Class : 1-10-25-20-1 fully connected architecture
class gen(nn.Module):

    def __init__(self):
        super(gen, self).__init__()
        self.fc1 = nn.Sequential(
                        nn.Linear(1,10),
                        nn.ReLU()
                      )
        self.fc2 = nn.Sequential(
                        nn.Linear(10,25),
                        nn.ReLU()
                      )
        self.fc3 = nn.Sequential(
                        nn.Linear(25,10),
                        nn.ReLU()
                      )
        self.out = nn.Linear(10,1)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.out(x)
        return x

# Discriminator Class : 1-10-5-1 fully connected architecture
class disc(nn.Module):

    def __init__(self):
        super(disc, self).__init__()
        self.fc1 = nn.Sequential(
                        nn.Linear(1,10),
                        nn.Tanh()
                      )
        self.fc2 = nn.Sequential(
                        nn.Linear(10,5),
                        nn.Tanh()
                      )
        self.out = nn.Sequential(
                        nn.Linear(5,1),
                        nn.Sigmoid()
                      )
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.out(x)
        return x

# Learning Rates: Discriminator's larger than generator's
gen_alpha = 0.05
disc_alpha = 0.06

def plotter(epo, GenNet, DiscNet):
    DiscNet = DiscNet.double()
    GenNet.eval()
    with torch.no_grad():
        outputs = GenNet(inputs)
    DiscNet.eval()
    xaxis = np.linspace(-3, 3, 100, dtype=np.double)
    x_data = 1/(0.2*np.sqrt(2*np.pi)) * np.exp(-(xaxis-2)**2/(2*0.2**2))
    out_mu = torch.mean(outputs).item()
    out_sigma = torch.std(outputs).item()
    x_values = 1/(out_sigma*np.sqrt(2*np.pi)) * np.exp(-(xaxis-out_mu)**2/(2*out_sigma**2))
    xaxis = xaxis[:,np.newaxis]
    disc_out = DiscNet(torch.from_numpy(xaxis))
    plt.plot(xaxis, x_data, 'b', label = 'Data dist')
    plt.plot(xaxis, disc_out.detach().numpy(), 'y', label = 'Discriminator')
    plt.plot(xaxis, x_values, 'k', label = 'Generated dist')
    in_counts, in_bins = np.histogram(inputs, bins=20)
    out_counts, out_bins = np.histogram(outputs, bins=20)
    plt.hist(in_bins[:-1], in_bins,  density = True, weights = in_counts, facecolor = 'green', label = 'Noise')
    plt.hist(out_bins[:-1], out_bins,  density = True, weights = out_counts, facecolor = 'red', label = 'Generated hist')
    plt.xlim(-3,3)
    plt.ylim(0,5)
    plt.title('Epoch:'+ str(epo))
    plt.legend(loc='upper left')
    plt.show()

def GANTrainer(GenNet, DiscNet):
    optimizer1 = optim.SGD(GenNet.parameters(), lr = gen_alpha)
    optimizer2 = optim.SGD(DiscNet.parameters(), lr = disc_alpha)
    iters = 0
    for epoch in range(100):
        GenNet = GenNet.float()
        DiscNet = DiscNet.float()
        GenNet.train()
        DiscNet.train()
        z_arr = torch.rand((64,1))*2 - 1
        for k in range(10):
            optimizer2.zero_grad()
            out = GenNet(z_arr)
            discloss = -1*torch.mean(torch.log(1 - DiscNet(out)) + torch.log(DiscNet(torch.randn((64,1))*0.2+2)))
            discloss.backward()
            optimizer2.step()
        optimizer1.zero_grad()
        out = GenNet(z_arr)
        genloss = -1*torch.mean(torch.log(DiscNet(out)))
        genloss.backward()
        optimizer1.step()
        iters += 1
        print("Epoch",epoch+1,"done...")
        if (epoch+1)%10 == 0:
            plotter(epoch+1, GenNet, DiscNet)
    print("Training complete!")

Gen = gen()
Disc = disc()

inputs = torch.rand((1000,1))*2 - 1

Gen = Gen.float()
Disc = Disc.float()
GANTrainer(Gen, Disc)

outputs = Gen(inputs)
out_mu = torch.mean(outputs).item()
out_sigma = torch.std(outputs).item()
print("Generated dist mean = "+str(out_mu))
print("Generated dist std. deviation = "+str(out_sigma))



# Downloading and Reading Data, Forming Train, Validation and Test Sets
tform = tf.ToTensor()
trainvaliddat = ds.MNIST('',download=True, train=True, transform=tform)
traindat,validdat = torch.utils.data.random_split(trainvaliddat,(50000,10000)) # random_split does a random shuffle before creating training and validation sets
trainwhole = torch.utils.data.DataLoader(traindat, batch_size = 50000)
trainld = torch.utils.data.DataLoader(traindat,batch_size=500)
validld = torch.utils.data.DataLoader(validdat,batch_size=1000)
tstdat = ds.MNIST('',download=True, train=False, transform=tform)
tstld = torch.utils.data.DataLoader(tstdat,batch_size=1000)
