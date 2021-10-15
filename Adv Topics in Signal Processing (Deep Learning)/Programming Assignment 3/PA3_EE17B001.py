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
params = {"text.color" : "y",
          "ytick.color" : "y",
          "xtick.color" : "y",
          "axes.labelcolor" : "y",
          "axes.edgecolor" : "y"}
plt.rcParams.update(params)
import math

# Downloading and Reading Data, Forming Train, Validation and Test Sets
tform = tf.ToTensor()
trainvaliddat = ds.MNIST('',download=True, train=True, transform=tform)
traindat,validdat = torch.utils.data.random_split(trainvaliddat,(50000,10000)) # random_split does a random shuffle before creating training and validation sets
trainld = torch.utils.data.DataLoader(traindat,batch_size=500)
validld = torch.utils.data.DataLoader(validdat,batch_size=1000)
tstdat = ds.MNIST('',download=True, train=False, transform=tform)
tstld = torch.utils.data.DataLoader(tstdat,batch_size=1000)

# Class to Produce Model Accepting Even the Network Type as Input (Can be Used for First 2 Questions)
class Model(nn.Module):
    def __init__(self, input_size, output_size, h_dim, n_layers, net, bidirect):
        super(Model, self).__init__()
        self.h_dim = h_dim
        self.n_layers = n_layers
        self.lstmflag = (net == nn.LSTM)
        self.net = net(input_size, h_dim, n_layers, batch_first=True, bidirectional=bidirect)
        self.fc = nn.Linear(h_dim*(1+bidirect), output_size)
        self.bidirect = bidirect
    
    def forward(self, x):
        batch_size = x.size(0)
        h = self.init_h(batch_size)
        c = self.init_c(batch_size)
        if self.lstmflag:
            out, (h, c) = self.net(x, (h, c))
        else:
            out, h = self.net(x, h)
        out = out.contiguous()[:,-1,:]
        out = self.fc(out)
        if self.lstmflag:
            return out, (h, c)
        else:
            return out, h 
    
    def init_h(self, batch_size):
        layers = self.n_layers*(1+self.bidirect)
        h = torch.zeros(layers, batch_size, self.h_dim)
        return h
    
    def init_c(self, batch_size):
        layers = self.n_layers*(1+self.bidirect)
        c = torch.zeros(layers, batch_size, self.h_dim)
        return c
        
        
# Function to Create an Instance of the Network with Hidden Layer Dimension set as 100
def ObjectCreator(net, lossfn, lr, bidirect):
    Net = Model(28, 10, 100, 1, net, bidirect)
    lstmflag = (net == nn.LSTM)
    criterion = lossfn()
    alpha = lr
    return Net, criterion, alpha, lstmflag
    
    
# Function to Train the Model
def Trainer(Net, lossfn, lrate, lstmflag, epochs, wd):
    optimizer = optim.Adam(Net.parameters(), lr=lrate)
    trainlosses = []; validlosses = []; validxaxis = []; validaccu = []
    iters = 0
    for i in range(epochs):
        Net.eval()
        count = 0; t = 0
        loss = 0; den = 0
        # Computing Validation Losses
        with torch.no_grad():
            for img, lbl in validld:
                img = img.view(-1, 28, 28)
                optimizer.zero_grad()
                if lstmflag:
                    out, (h, c) = Net(img)
                else:
                    out, h = Net(img)
                loss += lossfn(out,lbl)+wd*torch.norm(Net.fc.weight)
                den += 1
                val, ind = torch.max(out.data,1)
                count += ((ind == lbl).sum()).item()
                t += img.size(0)
                validlosses.append(loss/den)
                validxaxis.append(iters)
                validaccu.append(count/t*100)
        Net.train()
        for img, lbl in trainld:
            img = img.view(-1, 28, 28)
            optimizer.zero_grad()
            if lstmflag:
                out, (h, c) = Net(img)
            else:
                out, h = Net(img)
            loss = lossfn(out, lbl)
            loss.backward()
            optimizer.step()
            trainlosses.append(loss)
            iters+=1
        print("Epoch",i+1,"done...")
    plt.plot(range(iters),trainlosses, label = 'Training Error')
    plt.plot(validxaxis, validlosses, 'r', label = 'Validation Error')
    plt.title('Error vs Iterations, Regularisation Parameter '+str(wd), fontweight='bold', fontsize=15)
    plt.xlabel('Number of Iterations\u279d')
    plt.ylabel('Error\u279d')
    plt.legend()
    plt.show()
    plt.plot(validxaxis, validaccu, 'r')
    plt.title('Validation Accuracy vs Iterations', fontweight='bold', fontsize = 15)
    plt.xlabel('Number of Iterations\u279d')
    plt.ylabel('Accuracy\u279d')
    plt.show()
    return None
    
    
# Function to Compute Test Accuracy
def Tester(Net):
    Net.eval()
    with torch.no_grad():
        t = 0 ; count = 0
        for img, lbl in tstld:
            img = img.view(-1, 28, 28)
            out, _ = Net(img)
            val, ind = torch.max(out.data,1)
            count += ((ind == lbl).sum()).item()
            t += img.size(0)
    print('Test Accuracy of the Network = ', (count/t*100),'%')  
    
    
#Function to Select Random Test Samples and Show True vs Predicted Labels
def RandomTrueVsPredicted(Net, num_examples):
    rnd = np.ceil(abs(np.random.randn(num_examples))*1000)%1000
    pred = []; truth = []
    plt.figure()
    img, lbl = next(iter(tstld))
    for j in range (num_examples):
        pick=int(rnd[j])
        plt.subplot(1,num_examples,j+1)
        plt.imshow(img[pick][0].detach().numpy(), cmap = 'gray')
        plt.xticks([])
        plt.yticks([])
        out, _ = Net(img[pick][0].unsqueeze(0))
        truth.append(lbl[pick].item())
        pred.append(torch.argmax(func.softmax(out,dim=1)).item())
    plt.show()  
    print('True labels = ',truth,", Predicted labels = ",pred)
  
  
# Training Unregularised Vanilla RNN
RNN, LossFn, LearnRate, LSTMFlag = ObjectCreator(nn.RNN, lossfn = nn.CrossEntropyLoss, lr = 0.001, bidirect = False)
Trainer(RNN, LossFn, LearnRate, LSTMFlag, epochs = 10, wd = 0)

# Testing Unregularised Vanilla RNN
Tester(RNN)
RandomTrueVsPredicted(RNN, num_examples = 7)

# Training Regularised Vanilla RNN
rRNN, LossFn, LearnRate, LSTMFlag = ObjectCreator(nn.RNN, lossfn = nn.CrossEntropyLoss, lr = 0.001, bidirect = False)
Trainer(rRNN, LossFn, LearnRate, LSTMFlag, epochs = 10, wd = 0.003)

# Testing Unregularised Vanilla RNN
Tester(rRNN)

# Training Unregularised GRU
GRU, LossFn, LearnRate, LSTMFlag= ObjectCreator(nn.GRU, lossfn = nn.CrossEntropyLoss, lr = 0.001, bidirect = False)
Trainer(GRU, LossFn, LearnRate, LSTMFlag, epochs = 10, wd = 0)

# Testing Unregularised GRU
Tester(GRU)
RandomTrueVsPredicted(GRU, num_examples = 7)

# Training Regularised GRU
rGRU, LossFn, LearnRate, LSTMFlag= ObjectCreator(nn.GRU, lossfn = nn.CrossEntropyLoss, lr = 0.001, bidirect = False)
Trainer(rGRU, LossFn, LearnRate, LSTMFlag, epochs = 10, wd = 0.00001)

# Testing Regularised GRU
Tester(rGRU)

# Training Unregularised Bidirectional LSTM
BiLSTM, LossFn, LearnRate, LSTMFlag = ObjectCreator(nn.LSTM, lossfn = nn.CrossEntropyLoss, lr = 0.001, bidirect = True)
Trainer(BiLSTM, LossFn, LearnRate, LSTMFlag, epochs = 10, wd = 0)

# Testing Unregularised Bidirectional LSTM
Tester(BiLSTM)
RandomTrueVsPredicted(BiLSTM, num_examples = 7)

# Training Regularised Bidirectional LSTM
rBiLSTM, LossFn, LearnRate, LSTMFlag = ObjectCreator(nn.LSTM, lossfn = nn.CrossEntropyLoss, lr = 0.001, bidirect = True)
Trainer(rBiLSTM, LossFn, LearnRate, LSTMFlag, epochs = 10, wd = 0.00001)

# Testing Regularised Bidirectional LSTM
Tester(rBiLSTM)

# Function to Produce Training Samples for Q2
def trainsample2(L):
    rnd = np.ceil(abs(np.random.randn(L))*500)%500
    inp = torch.zeros(L,10)
    outp = torch.zeros(1,10)
    img, lbl = next(iter(trainld))
    for j in range(L):
        inp[j][lbl[int(rnd[j])]] = 1
    trg = inp[1]
    return (inp, trg)
    
# Function to Produce Testing Samples for Q2
def testsample2(L):
    rnd = np.ceil(abs(np.random.randn(L))*500)%500
    inp = torch.zeros(L,10)
    outp = torch.zeros(1,10)
    img, lbl = next(iter(tstld))
    for j in range(L):
        inp[j][lbl[int(rnd[j])]] = 1
    trg = inp[1]
    return (inp, trg)
    
# Function to Create the 3 Networks with Hidden Layer Dimensions 2, 5, 10 Respectively
def Q2_ObjectCreator(net1, net2, net3, bidirect1, bidirect2, bidirect3):
    Net1 = Model(10, 10, 2, 1, net1, bidirect1)
    Net2 = Model(10, 10, 5, 1, net2, bidirect2)
    Net3 = Model(10, 10, 10, 1, net3, bidirect3)
    lstmflag1 = (net1 == nn.LSTM)
    lstmflag2 = (net2 == nn.LSTM)
    lstmflag3 = (net3 == nn.LSTM)
    return Net1, Net2, Net3, lstmflag1, lstmflag2, lstmflag3
    
    
# Creation of Validation Set
validset = []
validset_size = 200
for k in range(validset_size):
    inp_seqlength = int(np.ceil(abs(np.random.randn())*8)%8)+3
    (inp, trg) = trainsample2(inp_seqlength)
    inp = inp.unsqueeze(0)
    lbl = torch.argmax(trg).unsqueeze(0)
    validset.append((inp, lbl))
    
    
# Creation of Training Set
trainset = []
trainset_size = 600
for k in range (trainset_size):
    inp_seqlength = int(np.ceil(abs(np.random.randn())*8)%8)+3
    (inp, trg) = trainsample2(inp_seqlength)
    inp = inp.unsqueeze(0)
    lbl = torch.argmax(trg).unsqueeze(0)
    trainset.append((inp,lbl))
    
    
# Function to Train Any One Network
def Q2_Trainer(Net, lossfn, lrate, lstmflag, epochs, trset, valset, hdim):
    optimizer = optim.Adam(Net.parameters(), lr=lrate)
    trainlosses = []; validlosses = []; validxaxis = []; validaccu = []
    iters = 0
    criterion = lossfn()
    print('Model with Hidden Layer Dimension =', hdim)
    print('***************************************')
    
    for i in range(epochs):
        Net.eval()
        count = 0; t = 0
        loss = 0; den = 0
        with torch.no_grad():
            for (inp, lbl) in validset:
                optimizer.zero_grad()                
                if lstmflag:
                    out, (h, c) = Net(inp)
                else:
                    out, h = Net(inp)
                loss += criterion(out, lbl)
                den += 1
                val, ind = torch.max(out.data, 1)
                count += (ind == lbl).item()
                t += 1
                validlosses.append(loss/den)
                validxaxis.append(iters)
                validaccu.append(count/t*100)
        Net.train()
        for inp, lbl in trainset:
            optimizer.zero_grad()
            if lstmflag:
                out, (h, c) = Net(inp)
            else:
                out, h = Net(inp)
            loss = criterion(out, lbl)
            loss.backward()
            optimizer.step()
            trainlosses.append(loss)
            iters+=1
        print("Epoch",i+1,"done...")
        
    plt.plot(range(iters),trainlosses, label = 'Training Error')
    plt.plot(validxaxis, validlosses, 'r', label = 'Validation Error')
    plt.title('Error vs Iterations: Hidden Layer Dimension '+str(hdim), fontweight='bold', fontsize=15)
    plt.xlabel('Number of Iterations\u279d')
    plt.ylabel('Error\u279d')
    plt.legend()
    plt.show()
    plt.plot(validxaxis, validaccu, 'r')
    plt.title('Validation Accuracy vs Iterations', fontweight='bold', fontsize = 15)
    plt.xlabel('Number of Iterations\u279d')
    plt.ylabel('Accuracy\u279d')
    plt.show()
    print('*******************************************************************************')
    
    
# Creation of Test Set
testset = []
testset_size = 200
for k in range (testset_size):
    inp_seqlength = int(np.ceil(abs(np.random.randn())*8)%8)+2
    (inp, trg) = testsample2(inp_seqlength)
    inp = inp.unsqueeze(0)
    lbl = torch.argmax(trg).unsqueeze(0)
    testset.append((inp, lbl))
    
 
# Function to Test Any One Network
def Q2_Tester(Net, testset):
    Net.eval()
    with torch.no_grad():
        t = 0 ; count = 0
        for (inp, lbl) in testset:
            out, _ = Net(inp)
            val, ind = torch.max(out.data, 1)
            count += (ind == lbl).item()
            t += 1
    print('Test Accuracy of the Model = ', (count/t*100),'%')
    
# Training All 3 Networks
LSTM1, LSTM2, LSTM3, LSTMFlag1, LSTMFlag2, LSTMFlag3 = Q2_ObjectCreator(net1 = nn.LSTM, net2 = nn.LSTM, net3 = nn.LSTM, bidirect1 = False, bidirect2 = False, bidirect3 = False)
Q2_Trainer(LSTM1, lossfn = nn.CrossEntropyLoss, lrate = 0.01, lstmflag = LSTMFlag1, epochs = 20, trset = trainset, valset = validset, hdim = 2)
Q2_Trainer(LSTM2, lossfn = nn.CrossEntropyLoss, lrate = 0.01, lstmflag = LSTMFlag2, epochs = 20, trset = trainset, valset = validset, hdim = 5)
Q2_Trainer(LSTM3, lossfn = nn.CrossEntropyLoss, lrate = 0.01, lstmflag = LSTMFlag3, epochs = 20, trset = trainset, valset = validset, hdim = 10)


# Testing All 3 Networks
Q2_Tester(LSTM1, testset)
Q2_Tester(LSTM2, testset)
Q2_Tester(LSTM3, testset)


# Generating 5 Samples Each for Input Sequenece Lengths Varying from 3 to 10, and Printing Predicted 2nd Digit for Each

num_samples = 5
print('Number of samples for each sequence length =', num_samples)
print()
print('Sequence\t\t\t\t\t\tPredicted Number at Position 2')
print('***************************************************************************************')
for j in range(3,11):
    print('Sample Sequence Length =', j)
    for k in range (num_samples):
        (inp, trg) = testsample2(j)
        inp = inp.unsqueeze(0)
        out, _ = LSTM3(inp)
        inp = inp.squeeze()
        seq = torch.argmax(inp,1)
        ans = torch.argmax(out.data,1)
        if (j<=6):
            print(seq.numpy(),"\t\t\t\t\t\t\t\t", ans.item())
        else:
            print(seq.numpy(),"\t\t\t\t\t\t\t", ans.item())
            
            
# Class to Generate Blueprint of Network for Q3
class Q3_LSTMModel(nn.Module):
    def __init__(self, input_size, output_size, h_dim, n_layers, bidirect):
        super(Q3_LSTMModel, self).__init__()
        self.h_dim = h_dim
        self.n_layers = n_layers
        self.lstm = nn.LSTM(input_size, h_dim, n_layers, batch_first=True, bidirectional=bidirect)   
        self.fc = nn.Linear(h_dim*(1+bidirect), output_size)
        self.sgmd = nn.Sigmoid()
        self.bidirect = bidirect
    
    def forward(self, x):
        batch_size = x.size(0)
        h, c = self.init_hc(batch_size)
        out, (h, c) = self.lstm(x, (h, c))
        out = self.fc(out)
        out = self.sgmd(out)
        return out, (h, c) 
    
    def init_hc(self, batch_size):
        layers = self.n_layers*(1+self.bidirect)
        h = torch.zeros(layers, batch_size, self.h_dim)
        c = torch.zeros(layers, batch_size, self.h_dim)
        return h, c
        
        
# Creation of an Instance of the Network with Preferred Loss Function
def Q3_ObjectCreator(lossfn, hidden_dim, bidirect):
    out_dim = 0
    if lossfn == nn.MSELoss:
        out_dim = 1
    elif lossfn == nn.CrossEntropyLoss:
        out_dim = 2
    LSTMQ3 = Q3_LSTMModel(2, out_dim, hidden_dim, 1, bidirect)
    return LSTMQ3, hidden_dim
    
    
# Generation of a Training Sample of 2 Input Binary Numbers and Their Sum
def Q3_Sample(inp_len):
    bin1 = ''
    bin2 = ''
    for k in range(inp_len):
        bin1 += str(int(np.floor(np.random.rand()*2)))
        bin2 += str(int(np.floor(np.random.rand()*2)))
    summ = int(bin1, 2) + int(bin2, 2)
    quo = 1
    binsumm = ''
    while summ>0:
        binsumm = str(summ%2)+binsumm
        summ = int(summ/2)
    # Prefixing 0's in case the length of the sum exceeds length of either number
    if len(binsumm)==inp_len+1:
        bin1 = '0'+bin1
        bin2 = '0'+bin2
    else:
        while(len(binsumm)<inp_len):
            binsumm = '0'+binsumm
    return(bin1, bin2, binsumm)
    
    
# Creation of Training Set
trainset3 = []
trainset3_size = 1200
for k in range(trainset3_size):
    bin1, bin2, binsumm = Q3_Sample(inp_len = 5)
    trainset3.append((bin1, bin2, binsumm))
    
    
# Creation of Test Set 
# (Validation Set has not been asked in this question)
testset3 = []
testset3_size = 100
for k in range(testset3_size):
    bin1, bin2, binsumm = Q3_Sample(inp_len = 5)
    testset3.append((bin1, bin2, binsumm))
    
    
# Function to Train the Network and Calculate Test Loss at the End of Each Epoch
def Q3_Trainer(Net, lossfn, lrate, epochs, trset, tstset, hdim):
    optimizer = optim.Adam(Net.parameters(), lr=lrate)
    trainlosses = []; testlosses = []; testxaxis = []; testaccu = []
    iters = 0
    criterion = lossfn()
    print('Model with Hidden Layer Dimension =', hdim)
    print('Loss Function:',lossfn)
    print('***************************************')
    itersv = 0
    for i in range(epochs):
        Net.eval()
        count = 0; t = 0
        loss = 0; den = 0
        # Calculation of test losses
        with torch.no_grad():
            for (bin1, bin2, binsum) in tstset:
                optimizer.zero_grad() 
                b1 = torch.zeros(1, len(bin1), 2)
                bsum = torch.zeros(len(bin1), 1, dtype = torch.float)
                # The following for loop takes the binary numbers in the order of LSB to MSB for inputs, as required
                for k in range(len(bin1)):
                    b1[:,k,0] = int(bin1[-(k+1)])
                    b1[:,k,1] = int(bin2[-(k+1)])
                    bsum[k,:] = int(binsum[-(k+1)])
                out, (h, c) = Net(b1)
                out = out.squeeze(0)
                ind = 0
                if lossfn == nn.MSELoss:
                    ind = torch.floor(out.data*2)
                elif lossfn == nn.CrossEntropyLoss:
                    bsum = bsum.squeeze(1).long()
                    val, ind = torch.max(out.data, 1)
                count += ((ind == bsum).sum()).item()
                loss += criterion(out, bsum)
                den += 1
                itersv+=1
                t += len(bin1)
                testlosses.append(loss/den)
                testxaxis.append(iters)
                testaccu.append(count/t*100)
        Net.train()
        for (bin1, bin2, binsum) in trset:
            optimizer.zero_grad()
            b1 = torch.zeros(1, len(bin1), 2)
            bsum = torch.zeros(len(bin1), 1, dtype = torch.float)
            for k in range(len(bin1)):
                b1[:,k,0] = int(bin1[-(k+1)])
                b1[:,k,1] = int(bin2[-(k+1)])
                bsum[k,:] = int(binsum[-(k+1)])
            out, (h, c) = Net(b1)
            out = out.squeeze(0)
            if lossfn == nn.CrossEntropyLoss:
                bsum = bsum.squeeze(1).long()
            loss = criterion(out, bsum)
            loss.backward()
            optimizer.step()
            trainlosses.append(loss)
            iters+=1
        print("Epoch",i+1,"done...")
        
    plt.plot(range(iters),trainlosses, label = 'Training Error')
    plt.plot(testxaxis,testlosses, label = 'Test Error')
    plt.title('Error vs Iterations: Hidden Layer Dimension '+str(hdim), fontweight='bold', fontsize=15)
    plt.xlabel('Number of Iterations\u279d')
    plt.ylabel('Error\u279d')
    plt.legend()
    plt.show()
    plt.plot(range(itersv), testaccu, 'r')
    plt.title('Test Bit-Accuracy vs Iterations', fontweight='bold', fontsize = 15)
    plt.xlabel('Number of Iterations\u279d')
    plt.ylabel('Bit-Accuracy\u279d')
    plt.show()
    print('Bit-Accuracy = ', testaccu[-1])
    print('*******************************************************************************')
    
    
# Function to Test the Network
def Q3_Tester(Net, testset, lossfn):
    Net.eval()
    with torch.no_grad():
        t = 0 ; c = 0
        for ((bin1, bin2, binsum)) in testset:
            b1 = torch.zeros(1, len(bin1), 2)
            bsum = torch.zeros(len(bin1), 1, dtype = torch.float)
            # The following for loop takes the binary numbers in the order of LSB to MSB for inputs, as required
            for k in range(len(bin1)):
                b1[:,k,0] = int(bin1[-(k+1)])
                b1[:,k,1] = int(bin2[-(k+1)])
                bsum[k,:] = int(binsum[-(k+1)])
            out, _ = Net(b1)
            out = out.squeeze(0)
            ind = 0
            if lossfn == nn.MSELoss:
                ind = torch.floor(out.data*2)
            elif lossfn == nn.CrossEntropyLoss:
                bsum = bsum.squeeze(1).long()
                val, ind = torch.max(out.data, 1)
            c += ((ind == bsum).sum()).item()
            t += len(bin1)
    return (c/t*100)
    
    
# Training of two Networks, with MSE Loss and Cross Entropy Loss Respectively, with Given Input Number Lengths (given in the training set creation) and Hidden Layer Dimension (given here below as a function parameter)
LSTM_Q3_1, hid_dim = Q3_ObjectCreator(lossfn = nn.MSELoss, hidden_dim = 5, bidirect = False)
LSTM_Q3_2, hid_dim = Q3_ObjectCreator(lossfn = nn.CrossEntropyLoss, hidden_dim = 5, bidirect = False)
Q3_Trainer(LSTM_Q3_1, nn.MSELoss, lrate = 0.01, epochs = 2, trset = trainset3, tstset = testset3, hdim = hid_dim)
Q3_Trainer(LSTM_Q3_2, nn.CrossEntropyLoss, lrate = 0.01, epochs = 2, trset = trainset3, tstset = testset3, hdim = hid_dim)


# Evaluation and Plotting of Bit-Accuracies for Various Input Lengths (After Training on a Fixed Input Length)
mse_bitaccu = []
ce_bitaccu = []
L_list = []
for k in range(1,21):
    testset3 = []
    testset3_size = 100
    for j in range(testset3_size):
        bin1, bin2, binsumm = Q3_Sample(inp_len = k)
        testset3.append((bin1, bin2, binsumm))
    mse_bitaccu.append(Q3_Tester(LSTM_Q3_1, testset = testset3, lossfn = nn.MSELoss))
    ce_bitaccu.append(Q3_Tester(LSTM_Q3_2, testset = testset3, lossfn = nn.CrossEntropyLoss))
    L_list.append(k)
plt.plot(L_list, mse_bitaccu, label = 'MSE Loss Bit-Accuracy')
plt.plot(L_list, ce_bitaccu, 'r', label = 'CE Loss Bit-Accuracy')
plt.title('Bit-Accuracies ', fontweight='bold', fontsize=15)
plt.xlabel('Length of Input Binary Sequences\u279d')
plt.xticks(range(21))
plt.ylabel('Bit-Accuracy\u279d')
plt.legend()
plt.show()


