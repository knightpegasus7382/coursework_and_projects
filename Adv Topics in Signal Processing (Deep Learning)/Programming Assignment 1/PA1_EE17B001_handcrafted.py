import numpy.random as rn
from numpy import *
import idx2numpy as idn
import matplotlib.pyplot as pl
from skimage.feature import hog
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier

def hogg(imgs):		# HOG implemeted from scikit-image
    feats = []
    for i in imgs:
        feat = hog(i, orientations=9, pixels_per_cell=(7,7),cells_per_block=(1,1), visualize=False)
        feats.append(feat)
    feats = array(feats)
    return (feats)

def ffbpa(w1,w2,w3,w4,yr,x,k1):		# Feedforward and backpropagation algorithms
	x=(x-mean(x))/(amax(x)-amin(x))
	xb=vstack((x,ones(k1)))
	z2=dot(w1,xb)
	a2=tanh(z2)						# tanh Activation
	ab2=vstack((a2,ones(k1)))
	z3=dot(w2,ab2)
	a3=tanh(z3)
	ab3=vstack((a3,ones(k1)))
	z4=dot(w3,ab3)
	a4=tanh(z4)
	ab4=vstack((a4,ones(k1)))
	z5=dot(w4,ab4)
	y=exp(z5)
	den=sum(y,axis=0)
	y=y/den
	del5=y-yr
	
	del4=dot(delete(transpose(w4), (-1), axis=0),del5)*(1-tanh(a4)**2)
	del3=dot(delete(transpose(w3), (-1), axis=0),del4)*(1-tanh(a3)**2)
	del2=dot(delete(transpose(w2), (-1), axis=0),del3)*(1-tanh(a2)**2)
		
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
	#L2 Regularisation Implemented:
	lmbda=0.001
	wi=wi-(alp*wi*lmbda/k1)
	wii=wii-(alp*wii*lmbda/k1)
	wiii=wiii-(alp*wiii*lmbda/k1)
	wiv=wiv-(alp*wiv*lmbda/k1)
	
	# Weight Updation:
	wi=wi-(alp/k1)*deljw1
	wii=wii-(alp/k1)*deljw2
	wiii=wiii-(alp/k1)*deljw3
	wiv=wiv-(alp/k1)*deljw4
	return [wi,wii,wiii,wiv]

def celoss(wi,wii,wiii,wiv,yr,y,k1):
	loss=-1*yr*log(y)
	loss=sum(loss)/k1
	lmbda=0.001
	loss=loss+(lmbda/(2*k1)*(sum(wi**2)+sum(wi**2)+sum(wi**2)+sum(wi**2)))
	return loss

def metrics(algo):		# Evaluation of Metrics
	global conf
	accuracy=sum(diag(conf))/sum(conf)
	pr=diag(conf)/sum(conf,axis=0)
	re=diag(conf)/sum(conf,axis=1)
	precision=mean(pr)
	recall=mean(re)
	f1=mean(2*pr*re/(pr+re))
	errorrates=1-((diag(conf)+(sum(conf)-sum(conf,axis=0)-sum(conf,axis=1)+diag(conf)))/(sum(conf)))
	avgerror=mean(errorrates)
	stddeverror=std(errorrates)
	print("...")
	print(algo)
	print("...")
	print("Confusion Matrix:")
	set_printoptions(suppress=True)
	print(conf)
	print("Average error rate = "+str(avgerror*100)+"%")
	print("Standard deviation of error rate = "+str(stddeverror*100)+"%")
	print('\n')
	print("Final accuracy on test set = "+str(accuracy*100)+"%")
	print("Final precision on test set = "+str(precision*100)+"%")
	print("Final recall on test set = "+str(recall*100)+"%")
	print("Final F1-score on test set = "+str(f1))
	print('\n')
	
trainimgs=idn.convert_from_file('train-images.idx3-ubyte')
trainlbls=idn.convert_from_file('train-labels.idx1-ubyte')
counter=array(range(60000))
trainys=zeros((10,60000))
trainys[trainlbls,counter]=1

tstimgs=idn.convert_from_file('t10k-images.idx3-ubyte')
tstlbls=idn.convert_from_file('t10k-labels.idx1-ubyte')
counter=array(range(10000))
tstys=zeros((10,10000))
tstys[tstlbls,counter]=1

# Extracting HOG Features from both train and test image datasets
trainfeats=hogg(trainimgs)
trainfeats=trainfeats.transpose()
tstfeats=hogg(tstimgs)
tstfeats=tstfeats.transpose()

# Glorot Initialisation
dl1=sqrt(6/244)
dl2=sqrt(6/160)
dl3=sqrt(6/90)
dl4=sqrt(6/40)
w1=(rn.rand(100, trainfeats.shape[0])-0.5)*2*dl1; b1=zeros(100)
wb1=column_stack((w1,b1))
w2=(rn.rand(50, 100)-0.5)*2*dl2; b2=zeros(50)
wb2=column_stack((w2,b2))
w3=(rn.rand(25, 50)-0.5)*2*dl3; b3=zeros(25)
wb3=column_stack((w3,b3))
w4=(rn.rand(10, 25)-0.5)*2*dl4; b4=zeros(10)
wb4=column_stack((w4,b4))
c=0
alpha=0.1	# Learning Rate

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
		if(p+64<=size(trainfeats,axis=1)):
			x64=trainfeats[:,p:p+64]
			y64=trainys[:,p:p+64]
			p=p+64
			k=64
			[del2,del3,del4,del5,delj1,delj2,delj3,delj4,y]=ffbpa(wb1,wb2,wb3,wb4,y64,x64,k)
			[wb1,wb2,wb3,wb4]=grd(wb1,wb2,wb3,wb4,alpha,delj1,delj2,delj3,delj4,k)
			trprog.append(celoss(wb1,wb2,wb3,wb4,y64,y,k))
		elif(p%64!=0):
			x64=zeros((trainfeats.shape[0],64))
			y64=zeros((10,64))
			l=size(trainys, axis=1)
			x64[:,0:l-p]=trainfeats[:,p:l]
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
			k=10000
			[del2,del3,del4,del5,delj1,delj2,delj3,delj4,y]=ffbpa(wb1,wb2,wb3,wb4,tstys,tstfeats,k)
			teprog.append(celoss(wb1,wb2,wb3,wb4,tstys,y,k))
		n+=1
	conf=zeros((10,10))		# Confusion matrix calculation
	k=10000
	[del2,del3,del4,del5,delj1,delj2,delj3,delj4,y]=ffbpa(wb1,wb2,wb3,wb4,tstys,tstfeats,k)
	predlbls=argmax(y, axis=0)
	add.at(conf,tuple([tstlbls, predlbls]),1)
	accuracy=sum(diag(conf))/sum(conf)
	c+=1
	epochiter.append(n)
	print("Epoch #"+str(c)+" done...")
	print("Accuracy on test set after epoch #"+str(c)+" = "+str(accuracy*100)+"%")
metrics("MLP with HOG Feature Extraction")

trainfeats=trainfeats.transpose()
trainlbls=trainlbls.transpose()
tstfeats=tstfeats.transpose()
tstlbls=tstlbls.transpose()

knn=KNeighborsClassifier(10)		# Implementation of K Nearest Neighbours Classifier using scikit-learn
knn.fit(trainfeats,trainlbls)
predknn=knn.predict(tstfeats)
conf.fill(0)
add.at(conf, [array(tuple(tstlbls)),array(tuple(predknn))], 1)
metrics("K Nearest Neighbours with HOG Feature Extraction")

sv=svm.LinearSVC()					# Implementation of Support Vector Machine Classifier using scikit-learn
sv.fit(trainfeats,trainlbls)
predsv=sv.predict(tstfeats)
conf.fill(0)
add.at(conf, [array(tuple(tstlbls)),array(tuple(predsv))], 1)
metrics("SVM with HOG Feature Extraction")

pl.figure(1)
pl.plot(range(len(trprog)), trprog, label='Training error')
pl.plot(200*array(range(len(teprog))), teprog, '-ro', label='Test error (plotted once in 200 iterations)')
pl.plot(array(epochiter), zeros(c), 'k^', label='Completion of epoch')
pl.xlabel('Number of iterations/minibatches \u279d', fontweight='bold')
pl.ylabel('Cross entropy loss on softmax probabilities \u279d', fontweight='bold')
pl.title('Training Progress of Neural Net', fontweight='bold')
pl.legend(loc='upper right')
pl.show()



	
	
	

