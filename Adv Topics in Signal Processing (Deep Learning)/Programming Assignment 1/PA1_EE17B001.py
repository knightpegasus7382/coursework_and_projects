import numpy.random as rn
from numpy import *
import idx2numpy as idn
import matplotlib.pyplot as pl

print("Which activation function do you wish to use? (1 for Sigmoid/ 2 for ReLU/ 3 for tanh)")
ch1=input()
if(ch1!='1' and ch1!='2' and ch1!='3'):
	print("Invalid choice of activation function!")
	exit()
print("Do you want a plot of percentage inactive neurons vs iterations? (y/n)")
ch5=input()
if(ch5!='y' and ch5!='n'):
	print("Invalid choice regarding inactive neurons plot!")
	exit()
print("Do you wish to add Gaussian noise in the forward pass or the backward pass or neither? (f/b/n)")
ch2=input()
if(ch2!='f' and ch2!='b' and ch2!='n'):
	print("Invalid choice regarding adding Gaussian noise!")
	exit()
print("Do you wish to augment your training data with noise-added copies? (y/n)")
ch3=input()
if(ch3!='y' and ch3!='n'):
	print("Invalid choice regarding data augmentation!")
	exit()
print("Do you wish to implement L2 regularisation in training? (y/n)")
ch4=input()
if(ch1!='1' and ch1!='2' and ch1!='3'):
	print("Invalid choice regarding using L2 regularisation!")
	exit()
	
def ffbpa(w1,w2,w3,w4,yr,x,k1):		# Feedforward and backpropagation algorithms
	x=(x-127.5)/255
	xb=vstack((x,ones(k1)))
	z2=dot(w1,xb)
	if(ch1=='1'):
		a2=1/(1+exp(-z2))		# Sigmoid Activation
	elif (ch1=='2'):
		a2=z2*(z2>=0)		# ReLU Activation
	elif(ch1=='3'):
		a2=tanh(z2)			# tanh Activation
	if(ch2=='f'):	
		a2=a2+rn.normal(0,0.05*mean(a2),shape(a2))		# Noise added in forward pass
	ab2=vstack((a2,ones(k1)))
	z3=dot(w2,ab2)
	if(ch1=='1'):
		a3=1/(1+exp(-z3))
	elif (ch1=='2'):
		a3=z3*(z3>=0)
	elif(ch1=='3'):
		a3=tanh(z3)
	if(ch2=='f'):	
		a3=a3+rn.normal(0,0.05*mean(a3),shape(a3))
	ab3=vstack((a3,ones(k1)))
	z4=dot(w3,ab3)
	if(ch1=='1'):
		a4=1/(1+exp(-z4))
	elif (ch1=='2'): 
		a4=z4*(z4>=0)
	elif(ch1=='3'):
		a4=tanh(z4)
	if(ch2=='f'):	
		a4=a4+rn.normal(0,0.05*mean(a4),shape(a4))
	ab4=vstack((a4,ones(k1)))
	z5=dot(w4,ab4)
	y=exp(z5)
	den=sum(y,axis=0)
	y=y/den
	del5=y-yr		# Derivative of the softmax wrt each output
	
	if(ch2=='b'):		# Noise added in the backward pass
		a4=a4+rn.normal(0,0.05*mean(a4),shape(a4))
		a3=a3+rn.normal(0,0.05*mean(a3),shape(a3))
		a2=a2+rn.normal(0,0.05*mean(a2),shape(a2))
	
	if(ch1=='1'):		# Backpropagation for sigmoid activation
		del4=dot(delete(transpose(w4), (-1), axis=0),del5)*a4*(1-a4)
		del3=dot(delete(transpose(w3), (-1), axis=0),del4)*a3*(1-a3)
		del2=dot(delete(transpose(w2), (-1), axis=0),del3)*a2*(1-a2)
		
	if(ch1=='2'):		# Backpropagation for ReLU activation
		del4=dot(delete(transpose(w4), (-1), axis=0),del5)*(z4>=0)
		del3=dot(delete(transpose(w3), (-1), axis=0),del4)*(z3>=0)
		del2=dot(delete(transpose(w2), (-1), axis=0),del3)*(z2>=0)
		
	if(ch1=='3'):		# Backpropagation for tanh activation
		del4=dot(delete(transpose(w4), (-1), axis=0),del5)*(1-tanh(a4)**2)
		del3=dot(delete(transpose(w3), (-1), axis=0),del4)*(1-tanh(a3)**2)
		del2=dot(delete(transpose(w2), (-1), axis=0),del3)*(1-tanh(a2)**2)
	
	# Calculation of the gradient steps	
	deljw4=dot(del5,transpose(a4)); deljb4=sum(del5,axis=1)
	deljwb4=column_stack((deljw4,deljb4))
	deljw3=dot(del4,transpose(a3)); deljb3=sum(del4,axis=1)
	deljwb3=column_stack((deljw3,deljb3))
	deljw2=dot(del3,transpose(a2)); deljb2=sum(del3,axis=1)
	deljwb2=column_stack((deljw2,deljb2))
	deljw1=dot(del2,transpose(x)); deljb1=sum(del2,axis=1)
	deljwb1=column_stack((deljw1,deljb1))
	return [del2,del3,del4,del5,deljwb1,deljwb2,deljwb3,deljwb4,y]

def grd(wi,wii,wiii,wiv,alp,deljw1,deljw2,deljw3,deljw4,k1):
	if(ch4=='y'):	# L2 Regularisation
		lmbda=0.01
		wi=wi-(alp*wi*lmbda/k1)
		wii=wii-(alp*wii*lmbda/k1)
		wiii=wiii-(alp*wiii*lmbda/k1)
		wiv=wiv-(alp*wiv*lmbda/k1)
	wi=wi-(alp/k1)*deljw1	# Weight updation step
	wii=wii-(alp/k1)*deljw2
	wiii=wiii-(alp/k1)*deljw3
	wiv=wiv-(alp/k1)*deljw4
	return [wi,wii,wiii,wiv]

def celoss(wi,wii,wiii,wiv,yr,y,k1):	# Cross-entropy loss
	loss=-1*yr*log(y)
	loss=sum(loss)/k1
	if(ch4=='y'):
		lmbda=0.01
		loss=loss+(lmbda/(2*k1)*(sum(wi**2)+sum(wi**2)+sum(wi**2)+sum(wi**2)))
	return loss

trainimgs=idn.convert_from_file('train-images.idx3-ubyte')
trainimgs=trainimgs.reshape(60000,784)
trainimgs=trainimgs.transpose()
trainlbls=idn.convert_from_file('train-labels.idx1-ubyte')
counter=array(range(60000))
trainys=zeros((10,60000))
trainys[trainlbls,counter]=1
if(ch3=='y'):	# Data Augmentation
	trainimgsnoisy=trainimgs+rn.normal(0,0.03*255,shape(trainimgs))	
	trainimgs=concatenate((trainimgs,trainimgsnoisy),axis=1)
	trainys=concatenate((trainys,trainys),axis=1)

tstimgs=idn.convert_from_file('t10k-images.idx3-ubyte')
tstimgs=tstimgs.reshape(10000,784)
tstimgs=tstimgs.transpose()
tstlbls=idn.convert_from_file('t10k-labels.idx1-ubyte')
counter=array(range(10000))
tstys=zeros((10,10000))
tstys[tstlbls,counter]=1

# Glorot Initialisation
dl1=sqrt(6/1284)
dl2=sqrt(6/750)
dl3=sqrt(6/350)
dl4=sqrt(6/110)
w1=(rn.rand(500, 784)-0.5)*2*dl1; b1=zeros(500)
wb1=column_stack((w1,b1))
w2=(rn.rand(250, 500)-0.5)*2*dl2; b2=zeros(250)
wb2=column_stack((w2,b2))
w3=(rn.rand(100, 250)-0.5)*2*dl3; b3=zeros(100)
wb3=column_stack((w3,b3))
w4=(rn.rand(10, 100)-0.5)*2*dl4; b4=zeros(10)
wb4=column_stack((w4,b4))
c=0
alpha=7		# Learning Rate

trprog=[]; teprog=[]
n=0
epochiter=[]
inact2=[]; inact3=[]; inact4=[]
print("Total # of epochs: 15")
while(c<15):
	p=0
	flag=0
	while(flag==0):
		k=0
		if(p+64<=size(trainimgs,axis=1)):
			x64=trainimgs[:,p:p+64]
			y64=trainys[:,p:p+64]
			p=p+64
			k=64
			[del2,del3,del4,del5,delj1,delj2,delj3,delj4,y]=ffbpa(wb1,wb2,wb3,wb4,y64,x64,k)
			[wb1,wb2,wb3,wb4]=grd(wb1,wb2,wb3,wb4,alpha,delj1,delj2,delj3,delj4,k)
			trprog.append(celoss(wb1,wb2,wb3,wb4,y64,y,k))
		elif(p%64!=0):
			x64=zeros((784,64))
			y64=zeros((10,64))
			l=size(trainys, axis=1)
			x64[:,0:l-p]=trainimgs[:,p:l]
			y64[:,0:l-p]=trainys[:,p:l]
			flag+=1
			x64=x64[:, ~all(y64 == 0, axis=0)]
			y64=y64[:, ~all(y64 == 0, axis=0)]
			k=size(x64, axis=1)
			[del2,del3,del4,del5,delj1,delj2,delj3,delj4,y]=ffbpa(wb1,wb2,wb3,wb4,y64,x64,k)
			[wb1,wb2,wb3,wb4]=grd(wb1,wb2,wb3,wb4,alpha,delj1,delj2,delj3,delj4,k)
			trprog.append(celoss(wb1,wb2,wb3,wb4,y64,y,k))
		elif(p%64==0):
			flag+=1
		if(n%200==0):
			inact2p=len(del2[del2<=10**-5])/size(del2)*100
			inact2.append(inact2p)
			inact3p=len(del3[del3<=10**-5])/size(del3)*100
			inact3.append(inact3p)
			inact4p=len(del4[del4<=10**-5])/size(del4)*100
			inact4.append(inact4p)
			k=10000
			[del2,del3,del4,del5,delj1,delj2,delj3,delj4,y]=ffbpa(wb1,wb2,wb3,wb4,tstys,tstimgs,k)
			teprog.append(celoss(wb1,wb2,wb3,wb4,tstys,y,k))
		n+=1
	conf=zeros((10,10))		# Confusion matrix calculation
	k=10000
	[del2,del3,del4,del5,delj1,delj2,delj3,delj4,y]=ffbpa(wb1,wb2,wb3,wb4,tstys,tstimgs,k)
	predlbls=argmax(y, axis=0)
	add.at(conf,tuple([tstlbls, predlbls]),1)
	accuracy=sum(diag(conf))/sum(conf)
	#print(conf)
	c+=1
	epochiter.append(n)
	print("Epoch #"+str(c)+" done...")
	print("Accuracy on test set after epoch #"+str(c)+" = "+str(accuracy*100)+"%")

# Evaluation of Metrics:
pr=diag(conf)/sum(conf,axis=0)
re=diag(conf)/sum(conf,axis=1)
precision=mean(pr)
recall=mean(re)
f1=mean(2*pr*re/(pr+re))
errorrates=1-((diag(conf)+(sum(conf)-sum(conf,axis=0)-sum(conf,axis=1)+diag(conf)))/(sum(conf)))
avgerror=mean(errorrates)
stddeverror=std(errorrates)
print("Confusion Matrix:")
set_printoptions(suppress=True)
print(conf)
print("...")
print("Average error rate = "+str(avgerror*100)+"%")
print("Standard deviation of error rate = "+str(stddeverror*100)+"%")
print('\n')
print("Final accuracy on test set = "+str(accuracy*100)+"%")
print("Final precision on test set = "+str(precision*100)+"%")
print("Final recall on test set = "+str(recall*100)+"%")
print("Final F1-score on test set = "+str(f1))

pl.figure(1)
pl.plot(range(len(trprog)), trprog, label='Training error')
pl.plot(200*array(range(len(teprog))), teprog, '-ro', label='Test error (plotted once in 200 iterations)')
pl.plot(array(epochiter), zeros(c), 'k^', label='Completion of epoch')
pl.xlabel('Number of iterations/minibatches \u279d', fontweight='bold')
pl.ylabel('Cross entropy loss on softmax probabilities \u279d', fontweight='bold')
pl.title('Training Progress of Neural Net', fontweight='bold')
pl.legend(loc='upper right')

if(ch5=='y'):
	pl.figure(2)
	pl.title('Percentages of Inactive Neurons ($p_{inact}$) in Hidden Layers', fontweight='bold')
	pl.xlabel('Number of iterations/minibatches \u279d', fontweight='bold')
	pl.ylabel('Percentage of inactive neurons \u279d', fontweight='bold')
	pl.plot(range(len(inact2)), inact2, '-r', label='$p_{inact}$ for hidden layer 1')
	pl.plot(range(len(inact3)), inact3, '-g', label='$p_{inact}$ for hidden layer 2')
	pl.plot(range(len(inact4)), inact4, '-b', label='$p_{inact}$ for hidden layer 3')
	pl.legend(loc='upper left')
pl.show()



	
	
	
