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

# Downloading and Reading Data, Forming Train, Validation and Test Sets
tform = tf.ToTensor()
trainvaliddat = ds.MNIST('',download=True, train=True, transform=tform)
traindat,validdat = torch.utils.data.random_split(trainvaliddat,(50000,10000)) # random_split does a random shuffle before creating training and validation sets
trainwhole = torch.utils.data.DataLoader(traindat, batch_size = 50000)
trainld = torch.utils.data.DataLoader(traindat,batch_size=500)
validld = torch.utils.data.DataLoader(validdat,batch_size=1000)
tstdat = ds.MNIST('',download=True, train=False, transform=tform)
tstld = torch.utils.data.DataLoader(tstdat,batch_size=1000)

Xinps = next(iter(trainwhole))[0]
Xinps_stretch = Xinps.view(Xinps.shape[0], -1)

k = 30
def PCA(X, k = k):
    X_mean = torch.mean(X,0)
    X = X - X_mean.expand_as(X)
    U, S, V = torch.svd(X)
    return torch.mm(U[:,:k], torch.diag(S)[:k,:k]), V

eigen30, V = PCA(Xinps_stretch)
pca_recon_stretch = torch.mm(eigen30, torch.t(V[:,:k]))
pca_recon = pca_recon_stretch.view(pca_recon.shape[0], 28, 28)

rnd = int(abs(np.random.randn())*500)%500
plt.subplot(1,2,1)
plt.imshow(Xinps[rnd].squeeze(), cmap = 'gray')
plt.subplot(1,2,2)
plt.imshow(pca_recon[rnd], cmap = 'gray')
plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
plt.suptitle("Q1: Result from first 30 eigenvalues of PCA", x = 0.55, fontsize = 20, fontweight = 'bold')
plt.show()
print("Reconstruction Error (MSE Error) = ", criterion(pca_recon[rnd], Xinps[rnd].squeeze()).item())

class autoencoder1(nn.Module):

    def __init__(self):
        super(autoencoder1, self).__init__()
        self.encfc1 = nn.Sequential(
                        nn.Linear(784,512),
                        nn.ReLU()
                      )
        self.encfc2 = nn.Sequential(
                        nn.Linear(512,256),
                        nn.ReLU()
                      )
        self.encfc3 = nn.Sequential(
                        nn.Linear(256,128),
                        nn.ReLU()
                      )
        self.encfc4 = nn.Sequential(
                        nn.Linear(128,30),
                        nn.ReLU()
                      )
        self.decfc1 = nn.Sequential(
                        nn.Linear(30,128),
                        nn.ReLU()
                      )
        self.decfc2 = nn.Sequential(
                        nn.Linear(128,256),
                        nn.ReLU()
                      )
        self.decfc3 = nn.Sequential(
                        nn.Linear(256,784),
                        nn.ReLU()
                      )
    def forward(self, x):
        k = x.shape[0]
        x = x.view(k,-1)
        x = self.encfc1(x)
        x = self.encfc2(x)
        x = self.encfc3(x)
        x = self.encfc4(x)
        x = self.decfc1(x)
        x = self.decfc2(x)
        x = self.decfc3(x)
        x = x.view(k,1,28,28)
        return x, 2

ae1 = autoencoder1()
criterion = nn.MSELoss()
alpha = 0.003

def Trainer(Net):
    optimizer = optim.Adam(Net.parameters(), lr = alpha)
    trainlosses = []; iters = 0
    for epoch in range(10):
        Net.train()
        for img, lbl in trainld:
            optimizer.zero_grad()
            out, _ = Net(img)
            loss = criterion(out, img)
            trainlosses.append(loss)
            loss.backward()
            optimizer.step()
            iters += 1
        print("Epoch",epoch+1,"done...")
    print("Training complete!")
    plt.plot(range(iters), trainlosses)
    plt.grid(True)
    plt.xlabel('Iterations\u279d')
    plt.ylabel('MSE Loss\u279d')
    plt.title('Convergence (Training Error Plot)', fontsize = 16, fontweight = 'bold')
    plt.show()

Trainer(ae1)

img, lbl = next(iter(trainld))
rnd = int(abs(np.random.randn())*500)%500
plt.figure()
plt.subplot(1,2,1)
plt.imshow(img[rnd].squeeze().detach().numpy(), cmap = 'gray')
out = ae1(img[rnd].unsqueeze(0))
plt.subplot(1,2,2)
plt.imshow(out.squeeze().detach().numpy(), cmap = 'gray')
plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
plt.suptitle("Q1: Result from Autoencoder of Q1", x = 0.55, fontsize = 20, fontweight = 'bold')
plt.show()
print("Reconstruction Error (MSE Error) = ", criterion(out, img[rnd].unsqueeze(0)).item())

class autoencoderstd(nn.Module):
    def __init__(self, x):
        super(autoencoderstd, self).__init__()
        self.encfc = nn.Sequential(
                        nn.Linear(784,x),
                        nn.ReLU()
                      )
        self.decfc = nn.Sequential(
                        nn.Linear(x,784),
                        nn.ReLU()
                      )
    def forward(self, x):
        k = x.shape[0]
        x = x.view(k,-1)
        x = self.encfc(x)
        l1pen = torch.norm(x,p=1)
        x = self.decfc(x)
        x = x.view(k,1,28,28)
        return x, l1pen

ae64 = autoencoderstd(x = 64)
ae128 = autoencoderstd(x = 128)
ae256 = autoencoderstd(x = 256)

Trainer(ae64)
Trainer(ae128)
Trainer(ae256)

rnd = int(abs(np.random.randn())*500)%500
pick=int(rnd)
plt.figure()
plt.subplot(1,5,1)
plt.imshow(img[pick].squeeze().detach().numpy(), cmap = 'gray')
out64, _ = ae64(img[pick].unsqueeze(0))
out128, _ = ae128(img[pick].unsqueeze(0))
out256, _ = ae256(img[pick].unsqueeze(0))
plt.subplot(1,5,3)
plt.imshow(out64.squeeze().detach().numpy(), cmap = 'gray')
plt.subplot(1,5,4)
plt.imshow(out128.squeeze().detach().numpy(), cmap = 'gray')
plt.subplot(1,5,5)
plt.imshow(out256.squeeze().detach().numpy(), cmap = 'gray')
plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
plt.suptitle("Q2: Reconstruction for Standard AE with x = 64, 128, 256 Respectively", x = 0.55, fontsize = 20, fontweight = 'bold')
plt.show()  

N = torch.randn(28,28)
plt.figure()
plt.subplot(1,5,1)
plt.imshow(N.detach().numpy(), cmap = 'gray')
outrand64, _ = ae64(N.unsqueeze(0).unsqueeze(0))
outrand128, _ = ae128(N.unsqueeze(0).unsqueeze(0))
outrand256, _ = ae256(N.unsqueeze(0).unsqueeze(0))
plt.subplot(1,5,3)
plt.imshow(outrand64.squeeze().detach().numpy(), cmap = 'gray')
plt.subplot(1,5,4)
plt.imshow(outrand128.squeeze().detach().numpy(), cmap = 'gray')
plt.subplot(1,5,5)
plt.imshow(outrand256.squeeze().detach().numpy(), cmap = 'gray')
plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
plt.suptitle("Q2: Corresponding Results for Random Noise Inputs", x = 0.55, fontsize = 20, fontweight = 'bold')
plt.show()

class autoencodersparse(nn.Module):
    def __init__(self, ):
        super(autoencodersparse, self).__init__()
        self.encfc = nn.Sequential(
                        nn.Linear(784,1200),
                        nn.ReLU()
                      )
        self.decfc = nn.Sequential(
                        nn.Linear(1200,784),
                        nn.ReLU()
                      )
    def forward(self, x):
        k = x.shape[0]
        x = x.view(k,-1)
        x = self.encfc(x)
        l1pen = torch.norm(x,p=1)
        x = self.decfc(x)
        x = x.view(k,1,28,28)
        return x, l1pen

sparseae = autoencodersparse()
alpha = 0.001

def SparseTrainer(Net, lmbd):
    optimizer = optim.Adam(Net.parameters(), lr = alpha)
    trainlosses = []; iters = 0
    for epoch in range(10):
        Net.train()
        for img, lbl in trainld:
            optimizer.zero_grad()
            out, l1pen = Net(img)
            loss = criterion(out, img) + lmbd*l1pen
            trainlosses.append(loss)
            loss.backward()
            optimizer.step()
            iters += 1
        print("Epoch",epoch+1,"done...")
    print("Training complete!")
    plt.plot(range(iters), trainlosses)
    plt.grid(True)
    plt.xlabel('Iterations\u279d')
    plt.ylabel('MSE Loss\u279d')
    plt.title('Convergence (Training Error Plot)', fontsize = 16, fontweight = 'bold')
    plt.show()

SparseTrainer(sparseae, lmbd = 1e-07)

rnd = int(abs(np.random.randn())*500)%500
plt.figure()
plt.subplot(1,2,1)
plt.imshow(img[rnd].squeeze().detach().numpy(), cmap = 'gray')
out, l1 = sparseae(img[rnd].unsqueeze(0))
plt.subplot(1,2,2)
plt.imshow(out.squeeze().detach().numpy(), cmap = 'gray')
plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
plt.suptitle("Result from Sparse AE", x = 0.55, fontsize = 20, fontweight = 'bold')
plt.show()
outstd64, l1std64 = ae64(img[rnd].unsqueeze(0))
outstd128, l1std128 = ae128(img[rnd].unsqueeze(0))
outstd256, l1std256 = ae256(img[rnd].unsqueeze(0))
print("Average value of activations from Sparse AE = ", l1.item()/1200)
print("Average value of activations from Standard AE64 = ", l1std64.item()/64)
print("Average value of activations from Standard AE128 = ", l1std128.item()/128)
print("Average value of activations from Standard AE256 = ", l1std256.item()/256)

for i in range(3):
    plt.subplot(1,3,i+1)
    plt.imshow(sparseae.state_dict()['encfc.0.weight'][i].view(28,28), cmap = 'gray')
plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
plt.suptitle("Weights of Sparse AE", x = 0.55, fontsize = 20, fontweight = 'bold')
plt.show()
for i in range(3):
    plt.subplot(3,3,i+1)
    plt.imshow(ae64.state_dict()['encfc.0.weight'][i].view(28,28), cmap = 'gray')
    plt.subplot(3,3,i+4)
    plt.imshow(ae128.state_dict()['encfc.0.weight'][i].view(28,28), cmap = 'gray')
    plt.subplot(3,3,i+7)
    plt.imshow(ae256.state_dict()['encfc.0.weight'][i].view(28,28), cmap = 'gray')
plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
plt.suptitle("Weights of Standard AEs (first row AE64, second row AE128, third row AE256)", x = 0.55, fontsize = 20, fontweight = 'bold')
plt.show()

def noiser(x, frac):
    batchsize = x.shape[0]
    x = x.squeeze()
    len_x = x.view(batchsize,-1).shape[1]
    num_zeros = int(frac*len_x)
    indices = np.array(sample(range(0,784), num_zeros))
    y = x.view(batchsize,-1).clone()
    y[:,indices] = 0
    y = y.view(batchsize,28,28).unsqueeze(1)
    return y

noisyimg03 = noiser(img[rnd], frac = 0.3)
noisyimg05 = noiser(img[rnd], frac = 0.5)
noisyimg08 = noiser(img[rnd], frac = 0.8)
noisyimg09 = noiser(img[rnd], frac = 0.9)

plt.subplot(1,5,1)
plt.imshow(noisyimg03.squeeze().detach().numpy(), cmap = 'gray')
noisy64, _ = ae64(noisyimg03.unsqueeze(0))
noisy128, _ = ae128(noisyimg03.unsqueeze(0))
noisy256, _ = ae256(noisyimg03.unsqueeze(0))
plt.subplot(1,5,3)
plt.imshow(noisy64.squeeze().detach().numpy(), cmap = 'gray')
plt.subplot(1,5,4)
plt.imshow(noisy128.squeeze().detach().numpy(), cmap = 'gray')
plt.subplot(1,5,5)
plt.imshow(noisy256.squeeze().detach().numpy(), cmap = 'gray')
plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
plt.suptitle("Noisy Images passed to Standard AEs, Noise Level = 0.3", x = 0.55, fontsize = 20, fontweight = 'bold')
plt.show()  

plt.subplot(1,5,1)
plt.imshow(noisyimg05.squeeze().detach().numpy(), cmap = 'gray')
noisy64, _ = ae64(noisyimg05.unsqueeze(0))
noisy128, _ = ae128(noisyimg05.unsqueeze(0))
noisy256, _ = ae256(noisyimg05.unsqueeze(0))
plt.subplot(1,5,3)
plt.imshow(noisy64.squeeze().detach().numpy(), cmap = 'gray')
plt.subplot(1,5,4)
plt.imshow(noisy128.squeeze().detach().numpy(), cmap = 'gray')
plt.subplot(1,5,5)
plt.imshow(noisy256.squeeze().detach().numpy(), cmap = 'gray')
plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
plt.suptitle("Noisy Images passed to Standard AEs, Noise Level = 0.5", x = 0.55, fontsize = 20, fontweight = 'bold')
plt.show()  

plt.subplot(1,5,1)
plt.imshow(noisyimg08.squeeze().detach().numpy(), cmap = 'gray')
noisy64, _ = ae64(noisyimg08.unsqueeze(0))
noisy128, _ = ae128(noisyimg08.unsqueeze(0))
noisy256, _ = ae256(noisyimg08.unsqueeze(0))
plt.subplot(1,5,3)
plt.imshow(noisy64.squeeze().detach().numpy(), cmap = 'gray')
plt.subplot(1,5,4)
plt.imshow(noisy128.squeeze().detach().numpy(), cmap = 'gray')
plt.subplot(1,5,5)
plt.imshow(noisy256.squeeze().detach().numpy(), cmap = 'gray')
plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
plt.suptitle("Noisy Images passed to Standard AEs, Noise Level = 0.8", x = 0.55, fontsize = 20, fontweight = 'bold')
plt.show()  

plt.subplot(1,5,1)
plt.imshow(noisyimg09.squeeze().detach().numpy(), cmap = 'gray')
noisy64, _ = ae64(noisyimg09.unsqueeze(0))
noisy128, _ = ae128(noisyimg09.unsqueeze(0))
noisy256, _ = ae256(noisyimg09.unsqueeze(0))
plt.subplot(1,5,3)
plt.imshow(noisy64.squeeze().detach().numpy(), cmap = 'gray')
plt.subplot(1,5,4)
plt.imshow(noisy128.squeeze().detach().numpy(), cmap = 'gray')
plt.subplot(1,5,5)
plt.imshow(noisy256.squeeze().detach().numpy(), cmap = 'gray')
plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
plt.suptitle("Noisy Images passed to Standard AEs, Noise Level = 0.9", x = 0.55, fontsize = 20, fontweight = 'bold')
plt.show()  

def DenoiseTrainer(Net, nfrac):
    optimizer = optim.Adam(Net.parameters(), lr = alpha)
    trainlosses = []; iters = 0
    for epoch in range(10):
        Net.train()
        for img, lbl in trainld:
            optimizer.zero_grad()
            noisy = noiser(img, nfrac)
            out, _ = Net(noisy)
            loss = criterion(out, img)
            trainlosses.append(loss)
            loss.backward()
            optimizer.step()
            iters += 1
        print("Epoch",epoch+1,"done...")
    print("Training complete!")
    plt.plot(range(iters), trainlosses)
    plt.grid(True)
    plt.xlabel('Iterations\u279d')
    plt.ylabel('MSE Loss\u279d')
    plt.title('Convergence (Training Error Plot)', fontsize = 16, fontweight = 'bold')
    plt.show()

alpha = 0.001
denoiseae = autoencoderstd(x = 1200)
DenoiseTrainer(denoiseae, nfrac = 0.5)

plt.subplot(1,2,1)
plt.imshow(noisyimg03.squeeze().detach().numpy(), cmap = 'gray')
plt.subplot(1,2,2)
plt.imshow(denoiseae(noisyimg03)[0].squeeze().detach().numpy(), cmap = 'gray')
plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
plt.suptitle("Output of Denoising AE (Trained on Noise Level = 0.5) to Input Noise Level = 0.3", x = 0.55, fontsize = 20, fontweight = 'bold')
plt.show()  

plt.subplot(1,2,1)
plt.imshow(noisyimg05.squeeze().detach().numpy(), cmap = 'gray')
plt.subplot(1,2,2)
plt.imshow(denoiseae(noisyimg05)[0].squeeze().detach().numpy(), cmap = 'gray')
plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
plt.suptitle("Output of Denoising AE (Trained on Noise Level = 0.5) to Input Noise Level = 0.5", x = 0.55, fontsize = 20, fontweight = 'bold')
plt.show()  

plt.subplot(1,2,1)
plt.imshow(noisyimg08.squeeze().detach().numpy(), cmap = 'gray')
plt.subplot(1,2,2)
plt.imshow(denoiseae(noisyimg08)[0].squeeze().detach().numpy(), cmap = 'gray')
plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
plt.suptitle("Output of Denoising AE (Trained on Noise Level = 0.5) to Input Noise Level = 0.8", x = 0.55, fontsize = 20, fontweight = 'bold')
plt.show()  

plt.subplot(1,2,1)
plt.imshow(noisyimg09.squeeze().detach().numpy(), cmap = 'gray')
plt.subplot(1,2,2)
plt.imshow(denoiseae(noisyimg09)[0].squeeze().detach().numpy(), cmap = 'gray')
plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
plt.suptitle("Output of Denoising AE (Trained on Noise Level = 0.5) to Input Noise Level = 0.9", x = 0.55, fontsize = 20, fontweight = 'bold')
plt.show()  

for i in range(3):
    plt.subplot(1,3,i+1)
    plt.imshow(denoiseae.state_dict()['encfc.0.weight'][i].view(28,28), cmap = 'gray')
plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
plt.suptitle("Weights of Denoise AE", x = 0.55, fontsize = 20, fontweight = 'bold')
plt.show()
for i in range(3):
    plt.subplot(3,3,i+1)
    plt.imshow(ae64.state_dict()['encfc.0.weight'][i].view(28,28), cmap = 'gray')
    plt.subplot(3,3,i+4)
    plt.imshow(ae128.state_dict()['encfc.0.weight'][i].view(28,28), cmap = 'gray')
    plt.subplot(3,3,i+7)
    plt.imshow(ae256.state_dict()['encfc.0.weight'][i].view(28,28), cmap = 'gray')
plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
plt.suptitle("Weights of Standard AEs (first row AE64, second row AE128, third row AE256)", x = 0.55, fontsize = 20, fontweight = 'bold')
plt.show()

class autoencoder_manifold(nn.Module):
    def __init__(self):
        super(autoencoder_manifold, self).__init__()
        self.encfc1 = nn.Sequential(
                        nn.Linear(784,64),
                        nn.ReLU()
                      )
        self.encfc2 = nn.Sequential(
                        nn.Linear(64,8),
                        nn.ReLU()
                      )
        self.decfc1 = nn.Sequential(
                        nn.Linear(8,64),
                        nn.ReLU()
                      )
        self.decfc2 = nn.Sequential(
                        nn.Linear(64,784),
                        nn.ReLU()
                      )
    def forward(self, x, testing):
        k = x.shape[0]
        x = x.view(k,-1)
        x = self.encfc1(x)
        x = self.encfc2(x)
        if testing == True:
            x = x + (torch.randn(x.shape)+x.mean())*torch.sqrt(x.var())*0.4
        x = self.decfc1(x)
        x = self.decfc2(x)
        x = x.view(k,1,28,28)
        return x

criterion = nn.MSELoss()
manifoldae = autoencoder_manifold()
alpha = 0.001

rnd = int(abs(np.random.randn())*500)%500
rndnoise = (torch.randn(img[rnd].shape)+0.5)*0.5
rndimg = img[rnd]+ rndnoise
plt.imshow(rndimg.squeeze().detach().numpy(), cmap = 'gray')
plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
plt.suptitle("MNIST Data Point Moved in Random Directions in 784-D Space", x = 0.55, fontsize = 20, fontweight = 'bold')
plt.show()

def ManifoldTrainer(Net):
    optimizer = optim.Adam(Net.parameters(), lr = alpha)
    trainlosses = []; iters = 0
    for epoch in range(10):
        Net.train()
        for img, lbl in trainld:
            optimizer.zero_grad()
            out = Net(img, testing = False)
            loss = criterion(out, img)
            trainlosses.append(loss)
            loss.backward()
            optimizer.step()
            iters += 1
        print("Epoch",epoch+1,"done...")
    print("Training complete!")
    plt.plot(range(iters), trainlosses)
    plt.grid(True)
    plt.xlabel('Iterations\u279d')
    plt.ylabel('MSE Loss\u279d')
    plt.title('Convergence (Training Error Plot)', fontsize = 16, fontweight = 'bold')
    plt.show()

ManifoldTrainer(manifoldae)

rnd = int(abs(np.random.randn())*500)%500
manifoldout = manifoldae(img[rnd],testing=True)
plt.subplot(1,2,1)
plt.imshow(img[rnd].squeeze().detach().numpy(), cmap = 'gray')
plt.subplot(1,2,2)
plt.imshow(manifoldout.squeeze().detach().numpy(), cmap = 'gray')
plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
plt.suptitle("Output Reconstructed after Moving Randomly in the Hidden Layer (Manifold)", x = 0.55, fontsize = 20, fontweight = 'bold')
plt.show()

class autoencoder_conv(nn.Module):
    def __init__(self):
        super(autoencoder_conv, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, stride = 1, padding = 1)
        self.conv2 = nn.Conv2d(8, 16, 3, stride = 1, padding = 1)
        self.conv3 = nn.Conv2d(16, 16, 3, stride = 1, padding = 1)
        self.maxpool = nn.MaxPool2d(2, stride = 2, return_indices = True)
        self.unpool = nn.MaxUnpool2d(2)
        self.deconv1 = nn.ConvTranspose2d(16, 16, 3, stride = 1, padding = 1)
        self.deconv2 = nn.ConvTranspose2d(8, 8, 3, stride = 1, padding = 1)
        self.deconv3 = nn.ConvTranspose2d(1, 1, 3, stride = 1, padding = 1)
        self.unpoolconv1 = nn.Conv2d(16, 16, 3, stride = 1, padding = 1)
        self.unpoolconv2 = nn.Conv2d(16, 8, 3, stride = 1, padding = 1)
        self.unpoolconv3 = nn.Conv2d(8, 1, 3, stride = 1, padding = 1)
        self.onlydeconv1 = nn.ConvTranspose2d(16, 16, 3, stride = 2)
        self.onlydeconv2 = nn.ConvTranspose2d(16, 8, 2, stride = 2)
        self.onlydeconv3 = nn.ConvTranspose2d(8, 1, 2, stride = 2)
    def forward(self, x, unpool, deconv):
        x = self.conv1(x)
        s1 = x.size()
        x, indices1 = self.maxpool(x)
        x = self.conv2(x)
        s2 = x.size()
        x, indices2 = self.maxpool(x)
        x = self.conv3(x)
        s3 = x.size()
        x, indices3 = self.maxpool(x)
        if unpool == True and deconv == True:
            x = self.unpool(x, indices3, output_size = s3)
            x = self.unpoolconv1(x)
            x = self.deconv1(x)
            x = self.unpool(x, indices2, output_size = s2)
            x = self.unpoolconv2(x)
            x = self.deconv2(x)
            x = self.unpool(x, indices1, output_size = s1)
            x = self.unpoolconv3(x)
            x = self.deconv3(x)
        if unpool == True and deconv == False:
            x = self.unpool(x, indices3, output_size = s3)
            x = self.unpoolconv1(x)
            x = self.unpool(x, indices2, output_size = s2)
            x = self.unpoolconv2(x)
            x = self.unpool(x, indices1, output_size = s1)
            x = self.unpoolconv3(x)
        if unpool == False and deconv == True:
            x = self.onlydeconv1(x)
            x = self.onlydeconv2(x)
            x = self.onlydeconv3(x)
        return x

unpoolae = autoencoder_conv()
unpooldeconvae = autoencoder_conv()
deconvae = autoencoder_conv()
alpha = 0.001
criterion = nn.MSELoss()

def ConvAETrainer(Net, unpool, deconv):
    optimizer = optim.Adam(Net.parameters(), lr = alpha)
    trainlosses = []; iters = 0
    for epoch in range(10):
        Net.train()
        for img, lbl in trainld:
            optimizer.zero_grad()
            out = Net(img, unpool, deconv)
            loss = criterion(out, img)
            trainlosses.append(loss)
            loss.backward()
            optimizer.step()
            iters += 1
        print("Epoch",epoch+1,"done...")
    print("Training complete!")
    return trainlosses, iters

trainloss, iters = ConvAETrainer(unpoolae, unpool = True, deconv = False)
trainlossunpool = trainloss

img, lbl = next(iter(trainld))
rnd = int(abs(np.random.randn())*500)%500
recon = unpoolae(img[rnd].unsqueeze(0), unpool = True, deconv = False)
plt.subplot(1,2,1)
plt.imshow(img[rnd].squeeze().detach().numpy(), cmap = 'gray')
plt.subplot(1,2,2)
plt.imshow(recon.squeeze().detach().numpy(), cmap = 'gray')
plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
plt.suptitle("Output of Convolutional AE with Only Unpooling", x = 0.55, fontsize = 20, fontweight = 'bold')
plt.show()
errorval = nn.MSELoss()
print("Reconstruction error (MSE Error) =", errorval(recon, img[rnd].unsqueeze(0)).item())

trainloss, iters = ConvAETrainer(unpooldeconvae, unpool = True, deconv = True)
trainlossesunpooldeconv = trainloss

recon = unpooldeconvae(img[rnd].unsqueeze(0), unpool = True, deconv = True)
plt.subplot(1,2,1)
plt.imshow(img[rnd].squeeze().detach().numpy(), cmap = 'gray')
plt.subplot(1,2,2)
plt.imshow(recon.squeeze().detach().numpy(), cmap = 'gray')
plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
plt.suptitle("Output of Convolutional AE with Unpooling + Deconvolution", x = 0.55, fontsize = 20, fontweight = 'bold')
plt.show()
errorval = nn.MSELoss()
print("Reconstruction error (MSE Error) =", errorval(recon, img[rnd].unsqueeze(0)).item())

trainloss, iters = ConvAETrainer(deconvae, unpool = False, deconv = True)
trainlossesdeconv = trainloss

recon = deconvae(img[rnd].unsqueeze(0), unpool = True, deconv = False)
plt.subplot(1,2,1)
plt.imshow(img[rnd].squeeze().detach().numpy(), cmap = 'gray')
plt.subplot(1,2,2)
plt.imshow(recon.squeeze().detach().numpy(), cmap = 'gray')
plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
plt.suptitle("Output of Convolutional AE with Only Deconvolution", x = 0.55, fontsize = 20, fontweight = 'bold')
plt.show()
errorval = nn.MSELoss()
print("Reconstruction error (MSE Error) =", errorval(recon, img[rnd].unsqueeze(0)).item())

plt.plot(range(1000), trainlossunpool, 'k', label = 'Unpool')
plt.plot(range(1000), trainlossesunpooldeconv, 'r', label = 'Unpool + Deconv')
plt.plot(range(1000), trainlossesdeconv, 'b', label = 'Deconv')
plt.legend()
plt.grid(True)
plt.xlabel('Iterations\u279d')
plt.ylabel('MSE Loss\u279d')
plt.title('Convergence (Training Error Plots)', fontsize = 16, fontweight = 'bold')
plt.show()

plt.subplot(1,3,1)
plt.imshow(unpoolae.state_dict()['unpoolconv1.weight'][0][0], cmap = 'gray')
plt.subplot(1,3,2)
plt.imshow(unpoolae.state_dict()['unpoolconv2.weight'][0][0], cmap = 'gray')
plt.subplot(1,3,3)
plt.imshow(unpoolae.state_dict()['unpoolconv3.weight'][0][0], cmap = 'gray')
plt.suptitle('Some Convolutional Filters of Only Unpooling AE (1 from each layer)', x= 0.55, fontsize = 16, fontweight = 'bold')
plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
plt.show()

plt.subplot(1,3,1)
plt.imshow(unpooldeconvae.state_dict()['deconv1.weight'][0][0], cmap = 'gray')
plt.subplot(1,3,2)
plt.imshow(unpooldeconvae.state_dict()['deconv2.weight'][0][0], cmap = 'gray')
plt.subplot(1,3,3)
plt.imshow(unpooldeconvae.state_dict()['deconv3.weight'][0][0], cmap = 'gray')
plt.suptitle('Some Deconvolutional Filters of Unpool + Deconv AE (1 from each layer)', fontsize = 16, fontweight = 'bold')
plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
plt.show()

plt.subplot(1,3,1)
plt.imshow(unpooldeconvae.state_dict()['onlydeconv1.weight'][0][0], cmap = 'gray')
plt.subplot(1,3,2)
plt.imshow(unpooldeconvae.state_dict()['onlydeconv2.weight'][0][0], cmap = 'gray')
plt.subplot(1,3,3)
plt.imshow(unpooldeconvae.state_dict()['onlydeconv3.weight'][0][0], cmap = 'gray')
plt.suptitle('Some Deconvolutional Filters of Only Deconv AE (1 from each layer)', fontsize = 16, fontweight = 'bold')
plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
plt.show()

