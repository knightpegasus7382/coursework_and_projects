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
import math

# Downloading and Reading Data, Forming Train, Validation and Test Sets
tform = tf.ToTensor()
trainvaliddat = ds.MNIST('',download=True, train=True, transform=tform)
traindat,validdat = torch.utils.data.random_split(trainvaliddat,(40000,20000))
trainld = torch.utils.data.DataLoader(traindat,batch_size=400)
validld = torch.utils.data.DataLoader(validdat,batch_size=1000)
tstdat = ds.MNIST('',download=True, train=False, transform=tform)
tstld = torch.utils.data.DataLoader(tstdat,batch_size=1000)

# CNN Class
class ConvNN1(nn.Module):

    def __init__(self):
        super(ConvNN1, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, stride = 1, padding = 1)
        self.maxp1 = nn.MaxPool2d(2, stride = 2)
        self.conv2 = nn.Conv2d(32, 32, 3, stride = 1, padding = 1)
        self.maxp2 = nn.MaxPool2d(2, stride = 2)
        self.fc1 = nn.Linear(32*7*7, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = func.relu(self.conv1(x))
        x = self.maxp1(x)
        x = func.relu(self.conv2(x))
        x = self.maxp2(x)
        x = x.view(-1, self.stretch(x))
        x = func.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def stretch(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
        
# CNN Object Creation
CNN1 = ConvNN1()
criterion = nn.CrossEntropyLoss()
alpha=0.01

# Training CNN1
optimizer = optim.Adam(CNN1.parameters(), lr=alpha)
trainlosses = []; validlosses = []; validxaxis = []; validaccu = []
iters = 0
for epoch in range(5):
  CNN1.eval()
  c=0; t=0
  loss = 0; den = 0
  with torch.no_grad():
    for img, lbl in validld:
      optimizer.zero_grad()
      out = CNN1(img)
      loss += criterion(out,lbl)
      den +=1
      val, ind = torch.max(out.data,1)
      c += ((ind == lbl).sum()).item()
      t += img.size(0)
    validlosses.append(loss/den)
    validxaxis.append(iters)
    validaccu.append(c/t*100)
  CNN1.train()
  for img, lbl in trainld:
    optimizer.zero_grad()
    out = CNN1(img)
    loss = criterion(out, lbl)
    loss.backward()
    optimizer.step()
    trainlosses.append(loss)
    iters+=1
  plt.show()
  print("Epoch",epoch+1,"done...")
plt.plot(range(iters),trainlosses, label = 'Training Error')
plt.plot(validxaxis, validlosses, 'r', label = 'Validation Error')
plt.title('Error vs Iterations (without Batch Normalisation)', fontweight='bold', fontsize=15)
plt.xlabel('Number of Iterations\u279d')
plt.ylabel('Error\u279d')
plt.legend()
plt.show()
plt.plot(validxaxis, validaccu, 'r')
plt.title('Validation accuracy vs Iterations', fontweight='bold', fontsize = 15)
plt.xlabel('Number of Iterations\u279d')
plt.ylabel('Accuracy\u279d')
plt.show()

# Test Accuracy of CNN1
CNN1.eval()
with torch.no_grad():
  t = 0 ; c = 0
  for img, lbl in tstld:
    out = CNN1(img)
    val, ind = torch.max(out.data,1)
    c += ((ind == lbl).sum()).item()
    t += img.size(0)
print('Test Accuracy of the CNN = ', (c/t*100),'%')  

# True vs Predicted Labels of CNN1 on Random Inputs 
# Random images picked from variable 'img', therefore running the test segments once will ensure that 'img' points to a subset of test images
rnd = np.ceil(abs(np.random.randn(6))*400)%400
pred = []; truth = []
plt.figure()
for j in range (6):
  pick=int(rnd[j])
  plt.subplot(1,6,j+1)
  plt.imshow(img[pick][0].detach().numpy(), cmap = 'gray')
  plt.xticks([])
  plt.yticks([])
  out = CNN1(img[pick].unsqueeze(0))
  truth.append(lbl[pick].item())
  pred.append(torch.argmax(func.softmax(out,dim=1)).item())
plt.show()  
print('True labels = ',truth,", Predicted labels = ",pred)
  
# CNN Class with Batch Normalisation
class ConvNN2(nn.Module):

    def __init__(self):
        super(ConvNN2, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, stride = 1, padding = 1)
        self.conv1_bn = nn.BatchNorm2d(32)
        self.maxp1 = nn.MaxPool2d(2, stride = 2)
        self.conv2 = nn.Conv2d(32, 32, 3, stride = 1, padding = 1)
        self.conv2_bn = nn.BatchNorm2d(32)
        self.maxp2 = nn.MaxPool2d(2, stride = 2)
        self.fc1 = nn.Linear(32*7*7, 500)
        self.fc1_bn = nn.BatchNorm1d(500)
        self.fc2 = nn.Linear(500, 10)
        self.fc2_bn = nn.BatchNorm1d(10)
    def forward(self, x):
        x = func.relu(self.conv1_bn(self.conv1(x)))
        x = self.maxp1(x)
        x = func.relu(self.conv2_bn(self.conv2(x)))
        x = self.maxp2(x)
        x = x.view(-1, self.stretch(x))
        x = func.relu(self.fc1_bn(self.fc1(x)))
        x = self.fc2_bn(self.fc2(x))
        return x

    def stretch(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
        
# CNN with Batch Normalisation Object Creation
CNN2 = ConvNN2()
criterion = nn.CrossEntropyLoss()
alpha=0.01

# Training CNN2
optimizer = optim.Adam(CNN2.parameters(), lr=alpha)
trainlosses = []; validlosses = []; validxaxis = []; validaccu = []
iters = 0
for epoch in range(5):
  CNN2.eval()
  c=0; t=0
  loss = 0; den = 0
  with torch.no_grad():
    for img, lbl in validld:
      optimizer.zero_grad()
      out = CNN2(img)
      loss += criterion(out,lbl)
      den +=1
      val, ind = torch.max(out.data,1)
      c += ((ind == lbl).sum()).item()
      t += img.size(0)
    validlosses.append(loss/den)
    validxaxis.append(iters)
    validaccu.append(c/t*100)
  CNN2.train()
  for img, lbl in trainld:
    optimizer.zero_grad()
    out = CNN2(img)
    loss = criterion(out, lbl)
    loss.backward()
    optimizer.step()
    trainlosses.append(loss)
    iters+=1
  print("Epoch",epoch+1,"done...")
plt.plot(range(iters),trainlosses, label = 'Training Error')
plt.plot(validxaxis, validlosses, 'r', label = 'Validation Error')
plt.title('Error vs Iterations (with Batch Normalisation)', fontweight='bold', fontsize=15)
plt.xlabel('Number of Iterations\u279d')
plt.ylabel('Error\u279d')
plt.legend()
plt.show()
plt.plot(validxaxis, validaccu, 'r')
plt.title('Validation accuracy vs Iterations', fontweight='bold', fontsize = 15)
plt.xlabel('Number of Iterations\u279d')
plt.ylabel('Accuracy\u279d')
plt.show()

# Test Accuracy of CNN2
CNN2.eval()
with torch.no_grad():
  t = 0 ; c = 0
  for img, lbl in tstld:
    out = CNN2(img)
    val, ind = torch.max(out.data,1)
    c += ((ind == lbl).sum()).item()
    t += img.size(0)
print('Test Accuracy of the CNN with Batch Normalisation = ', (c/t*100),'%') 

# conv1 Layer Filters
cnv1 = CNN1.conv1.weight.data.numpy()
fig,figs = plt.subplots()
for i in range(len(cnv1)):
  figs = plt.subplot(4, len(cnv1)/4,i+1)
  figs.set_xticks([])
  figs.set_yticks([])
  whole=figs.imshow(cnv1[i][0], cmap='gray')
fig.suptitle('Conv1 Layer Filters', fontweight='bold', fontsize=25)
fig.subplots_adjust(right=1)
cbaxis = fig.add_axes([1.05, 0.15, 0.03, 0.7])
fig.colorbar(whole, cbaxis)
plt.show()

#conv2 Layer Filters
cnv2 = CNN1.conv2.weight.data.numpy()
fig,figs = plt.subplots()
for i in range(len(cnv2)):
  figs = plt.subplot(4, len(cnv2)/4,i+1)
  whole=figs.imshow(cnv2[i][0], cmap='gray')
  figs.set_xticks([])
  figs.set_yticks([])
fig.suptitle('Conv2 Layer Filters', fontweight='bold', fontsize=25)
fig.subplots_adjust(right=1)
cbaxis = fig.add_axes([1.05, 0.15, 0.03, 0.7])
fig.colorbar(whole, cbaxis)
plt.show()

# Activations of Convolutional Layers
# In order to generate activations of the layers, a random input needs to be given.
# Random images picked from variable 'img', therefore running the test segments once will ensure that 'img' points to a subset of test images
rndimg = img[math.ceil(abs(np.random.randn())*400)%400]
rndimg = rndimg.unsqueeze(0)
funct=nn.ReLU()
cnv1 = CNN1.maxp1(funct(CNN1.conv1(rndimg))).data.numpy()
plt.figure()
for i in range(len(cnv1[0])):
  plt.subplot(4, len(cnv1[0])/4,i+1)
  plt.imshow(cnv1[0][i], cmap='gray')
  plt.xticks([])
  plt.yticks([])
plt.suptitle("Activations of Conv1:", fontweight='bold', fontsize=20)
plt.show()

cnv2 = funct(CNN1.maxp2(CNN1.conv2(funct(CNN1.maxp1(CNN1.conv1(rndimg)))))).data.numpy()
plt.figure()
for i in range(len(cnv2[0])):
  plt.subplot(4, len(cnv2[0])/4,i+1)
  plt.imshow(cnv2[0][i], cmap='gray')
  plt.xticks([])
  plt.yticks([])
plt.suptitle("Activations of Conv2:", fontweight='bold', fontsize=20)
plt.show()

# Occlusion
# Random images picked from variable 'img', therefore running the test segments once will ensure that 'img' points to a subset of test images
rnd = np.ceil(abs(np.random.randn(10))*400)%400
rndimgs = img[rnd]
rndlbls = lbl[rnd]
lblclones = rndlbls.clone().numpy()

rowinds = np.arange(10)
colinds = lblclones

occluder = np.zeros((4,4))+0.15
probgrids = np.zeros((10,1,7,7))
for i in range(7):
  for j in range(7):
    rndclones=rndimgs.clone().numpy()
    rndclones[:,:,4*i:4*(i+1),4*j:4*(j+1)] = occluder
    rndocc = torch.from_numpy(rndclones)
    rndout = func.softmax(CNN1(rndocc), dim=1)
    probs = rndout.detach().numpy()[rowinds,colinds]
    probgrids[:,:,i,j] = probs.reshape(10,1)
probgrids = torch.from_numpy(probgrids)
rndclones = rndimgs.clone()

plt.figure(figsize=(6.4,2.4))
for i in range(10):
  ax1 = plt.subplot(2,10,i+1)
  ax1.imshow(rndclones[i][0], cmap='gray')
  ax1.set_xticks([])
  ax1.set_yticks([])
plt.suptitle('10 Random Images:', fontweight='bold', fontsize='18')
plt.show()
plt.figure(figsize=(6.4,2.4))
for i in range(10):
  ax2 = plt.subplot(2,10,i+1)
  ax2.imshow(probgrids[i][0], cmap='gray')
  ax2.set_xticks([])
  ax2.set_yticks([])
plt.suptitle('Probability wrt Patch Position (Black: P = 0, White: P= 1):', fontweight='bold', fontsize='10')
plt.show()

# Non-Targeted Attack
optimizer = optim.Adam(CNN2.parameters(), lr=alpha)
plt.figure()
plt.tight_layout()
step_size = 1
predictions = []
confidences = []
losseslist = []
for j in range (10):
  losslist = []
  X = torch.randn(1,1,28,28)+0.5
  X.requires_grad=True
  for epoch in range(1000):
    optimizer.zero_grad()
    X.requires_grad=True
    loss = CNN2(X)[:,j]
    loss.backward()
    X=torch.from_numpy(((X+step_size*X.grad).detach().numpy()))
    losslist.append(loss)
  losseslist.append(losslist)
  plt.subplot(2,5,j+1)
  plt.imshow(X[0,0,:,:].detach().numpy(), cmap='gray')
  plt.xticks([])
  plt.yticks([])
  confidences.append(100*torch.max(func.softmax(CNN2(X),dim=1)).item())
  predictions.append(torch.argmax(func.softmax(CNN2(X),dim=1)).item())
plt.suptitle('Non-Targeted Attack Adversarial Images (from 0 to 9)', fontweight='bold', fontsize = 18)
plt.show()
print("Predictions =")
print(predictions)
print("Confidences =")
print(confidences)

# Cost Function Evolution During Non-Targeted Attack
plt.figure()
j=0
for i in losseslist:
  plt.plot(range(len(i)),i, label=j)
  j+=1
plt.legend()
plt.title('Cost function for each MNIST Class', fontweight='bold', fontsize=20)
plt.xlabel('Epochs\u279d', fontsize=14)
plt.ylabel('Loss\u279d', fontsize=14)
plt.show()

# Creating a Tensor Containing the Unique Images 0-9 Exactly Once
indices=[]
for j in range(10):
  msk = torch.zeros(1000)+j
  indx = torch.nonzero(lbl == msk.long())[0]
  indices.append(indx)
distinctimgs=img[np.array(indices)]

# Targeted Attack
optimizer = optim.Adam(CNN2.parameters(), lr=alpha)
msel = nn.MSELoss()
trg = 6
beta = 180
step_size = 0.03  # beta and/or step_size need to be altered for each target class in order to get best output
plt.figure()
confidences = []
predictions = []
for j in range (10):
  X = torch.randn(1,1,28,28)+0.5
  X.requires_grad=True
  for epoch in range(1000):
    optimizer.zero_grad()
    X.requires_grad=True
    loss = CNN2(X)[:,trg] - beta*msel(X,distinctimgs[j].unsqueeze(0))
    loss.backward()
    X=torch.from_numpy(((X+step_size*X.grad).detach().numpy()))
  plt.subplot(2,5,j+1)
  plt.imshow(X[0,0,:,:].detach().numpy(), cmap='gray')
  plt.xticks([])
  plt.yticks([])
  confidences.append(100*torch.max(func.softmax(CNN2(X),dim=1)).item())
  predictions.append(torch.argmax(func.softmax(CNN2(X),dim=1)).item())
plt.suptitle('Targeted-Attack Generated Images for Each Target Image (Target Class '+str(trg)+')', fontweight='bold')
plt.show()
print("Predictions =")
print(predictions)
print("Confidences =")
print(confidences)

# Targeted Attack: Adding Noise, Keeping Target Class Fixed
optimizer = optim.Adam(CNN2.parameters(), lr=alpha)
target_class = 3
step_size = 0.004  # step_size need to be altered for each target class in order to get best output
plt.figure()
confidences = []
predictions = []
noises = torch.zeros(10,1,28,28)
for j in range (10):
  X = distinctimgs[j]
  N = torch.zeros(1,1,28,28)
  X.requires_grad=True
  for epoch in range(1000):
    optimizer.zero_grad()
    N.requires_grad=True
    XN=X+N
    loss = CNN2(XN)[:,target_class]
    loss.backward()
    N=torch.from_numpy(((N+step_size*N.grad).detach().numpy()))
  noises[j]=N
  plt.subplot(2,5,j+1)
  plt.imshow(XN[0,0,:,:].detach().numpy(), cmap='gray')
  plt.xticks([])
  plt.yticks([])
  confidences.append(100*torch.max(func.softmax(CNN2(XN),dim=1)).item())
  predictions.append(torch.argmax(func.softmax(CNN2(XN),dim=1)).item())
plt.suptitle('Targeted-Attack Generated Images for Each Original Class (Target Class '+str(target_class)+')', fontweight='bold')
plt.show()
plt.figure()
for j in range (10):
  plt.subplot(2,5,j+1)
  plt.imshow(noises[j,0,:,:].detach().numpy(), cmap='gray')
  plt.xticks([])
  plt.yticks([])
plt.suptitle('Corresponding Noises for Each Original Class (Target Class '+str(target_class)+')', fontweight='bold')
plt.show()
print("Predictions =")
print(predictions)
print("Confidences =")
print(confidences)

# Targeted Attack: Adding Noise, Keeping Original Class Fixed
optimizer = optim.Adam(CNN2.parameters(), lr=alpha)
original_class = 5
trg = np.arange(10)
step_size = 0.0014  # step_size need to be altered for each target class in order to get best output
plt.figure()
confidences = []
predictions = []
noises = torch.zeros(10,1,28,28)
for j in range (10):
  X = distinctimgs[original_class]
  N = torch.zeros(1,1,28,28)
  X.requires_grad=True
  for epoch in range(1000):
    optimizer.zero_grad()
    N.requires_grad=True
    XN=X+N
    loss = CNN2(XN)[:,trg[j]]
    loss.backward()
    N=torch.from_numpy(((N+step_size*N.grad).detach().numpy()))
  noises[j]=N
  plt.subplot(2,5,j+1)
  plt.imshow(XN[0,0,:,:].detach().numpy(), cmap='gray')
  plt.xticks([])
  plt.yticks([])
  confidences.append(100*torch.max(func.softmax(CNN2(XN),dim=1)).item())
  predictions.append(torch.argmax(func.softmax(CNN2(XN),dim=1)).item())
plt.suptitle('Targeted-Attack Generated Images for Each Target Class (Original Class ' + str(original_class)+')', fontweight='bold')
plt.show()
plt.figure()
for j in range (10):
  plt.subplot(2,5,j+1)
  plt.imshow(noises[j,0,:,:].detach().numpy(), cmap='gray')
  plt.xticks([])
  plt.yticks([])
plt.suptitle('Corresponding Noises for Each Target Class (Original Class ' + str(original_class)+')', fontweight='bold')
plt.show()
print("Predictions =")
print(predictions)
print("Confidences =")
print(confidences)

# 10 Test Examples + Adversarial Noise Matrices: True vs Predicted Labels
# This segment should be run after running the previous one
original_class = 5
fixedinds = []
msk = torch.zeros(1000) + original_class
indx = torch.nonzero(lbl == msk.long())[0:6]  # 6 images selected from original class
fixedinds.extend(indx)
rnd = math.ceil(abs(np.random.randn())*100)%100
indx = torch.nonzero(lbl != msk.long())[rnd:rnd+4]  # 4 from other classes
fixedinds.extend(indx)
fixedinds = np.array(fixedinds)
fixedimgs=img[fixedinds]
for i in range(10):
  X1 = fixedimgs[i]
  predictions = []
  for j in range (10):
    XN1 = X1 + noises[j]
    predictions.append(torch.argmax(func.softmax(CNN2(XN1.unsqueeze(0)),dim=1)).item())
  print("Test example #",i+1)
  print("Adding all the 10 noise matrices, we get:")
  print("True class = ",lbl[fixedinds[i]].item())
  print("Predicted classes = ", predictions)
  print()
  
