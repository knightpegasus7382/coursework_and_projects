#########################################################
#					ASSIGNMENT 4						#
#########################################################
#	Author: Akash Reddy A								#
#	Roll No: EE17B001									#
#########################################################

import sys
import numpy as np
from pylab import *
import scipy.integrate as scint

def eexp(i):
	return np.exp(i)
def coscos(i):
	return np.cos(np.cos(i))
def eexpu(i, k):
	return np.exp(i)*np.cos(k*i)
def coscosu(i, k):
	return np.cos(np.cos(i))*np.cos(k*i)
def eexpv(i, k):
	return np.exp(i)*np.sin(k*i)
def coscosv(i, k):
	return np.cos(np.cos(i))*np.sin(k*i)
	
x=np.linspace(-2*pi,4*pi, 1200)
epoints=eexp(x)
coscospoints=coscos(x)
fourierexp=eexp(x%(2*pi))
fouriercoscos=coscos(x%(2*pi))

figure('Figure 1')
semilogy(x, epoints, label='True function')
semilogy(x, fourierexp, label='Expected function from Fourier Series')
xlabel('x \u279d', size=15, fontstyle='italic')
ylabel('e\u02E3 \u279D', size=15, fontstyle='italic')
title("Q1. Semilog(y) Plot of e\u02E3 vs x")
legend(loc='upper left')
grid(True)
show()
figure('Figure 2')
plot(x,coscospoints, label='True function')
plot(x, fouriercoscos, label='Expected function from Fourier Series')
xlabel('x \u279d', size=15, fontstyle='italic')
ylabel('cos(cos(x)) \u279D', size=15, fontstyle='italic')
title("Q1. Plot of cos(cos(x)) vs x")
legend(loc='upper left')
grid(True)
show()

# Functions to generate the a and b Fourier Coefficients after taking the function as a parameter:
def a_list(fna):
	a=[list(scint.quad(fna,0,2*pi,args=(n))) for n in range(26)]
	a=1/pi*array([a[i][0] for i in range(len(a))])
	a[0]=a[0]/2
	return(a)
def b_list(fnb):
	b=[list(scint.quad(fnb,0,2*pi,args=(n))) for n in range(26)]
	b.remove(b[0])
	b=1/pi*array([b[i][0] for i in range(len(b))])
	return(b)
	
# Fourier Coefficients generated for the two functions exp(x) and cos(cosx):
expa=a_list(eexpu)
expb=b_list(eexpv)
coscosa=a_list(coscosu)
coscosb=b_list(coscosv)

expco = [None]*(len(expa)+len(expb))
coscosco = [None]*(len(coscosa)+len(coscosb))

# The Fourier Coefficients are arranged in an a-b-a-b alternating fashion:
expco[0] = expa[0]
expco[1::2] = expa[1:]
expco[2::2] = expb

coscosco[0] = coscosa[0]
coscosco[1::2] = coscosa[1:]
coscosco[2::2] = coscosb

xaxis_n=[int((m+1)/2) for m in range(51)]

# The magnitudes of the coefficients obtained through integration taken into new arrays:
coscoscoabs=[abs(m) for m in coscosco]
expcoabs=[abs(m) for m in expco]

x=linspace(0,2*pi,401)
x=x[:-1]
b1=eexp(x)
b2=coscos(x)
A=zeros((400,51))
A[:,0]=1
for k in range(1,26):
	A[:,2*k-1]=np.cos(k*x)
	A[:,2*k]=np.sin(k*x)
c1=lstsq(A,b1,rcond=None)[0]
c2=lstsq(A,b2,rcond=None)[0]

# The magnitudes of the estimated coefficients taken into new arrays:
c1abs=[abs(m) for m in c1]
c2abs=[abs(m) for m in c2]

figure('Figure 3')
semilogy(xaxis_n, expcoabs, 'ro', label='Coefficients through integration')
semilogy(xaxis_n, c1abs,'go', label='Coefficients through estimation')
xlabel('n \u279d', fontstyle='italic')
ylabel('Coefficients of f\u2081(x) = exp(x) \u279D', fontstyle='italic')
title("Q3. Semilog(y) Plot of Fourier coefficients of exp(x) vs n")
legend(loc='upper right')
grid(True)
show()
figure('Figure 4')
loglog(xaxis_n, expcoabs, 'ro',label='Coefficients through integration')
loglog(xaxis_n, c1abs, 'go',label='Coefficients through estimation')
xlabel('n \u279d', fontstyle='italic')
ylabel('Coefficients of f\u2081(x) = exp(x) \u279D', fontstyle='italic')
title("Q3. Log-log plot of Fourier coefficients of exp(x) vs n")
legend(loc='lower left')
grid(True)
show()
figure('Figure 5')
semilogy(xaxis_n, coscoscoabs, 'ro', label='Coefficients through integration')
semilogy(xaxis_n, c2abs, 'go', label='Coefficients through estimation')
xlabel('n \u279d', fontstyle='italic')
ylabel('Coefficients of f\u2082(x) = cos(cos(x)) \u279D', fontstyle='italic')
title("Q3. Semilog(y) plot of Fourier coefficients of cos(cos(x)) vs n")
legend(loc='upper right')
grid(True)
show()
figure('Figure 6')
loglog(xaxis_n, coscoscoabs, 'ro', label='Coefficients through integration')
loglog(xaxis_n, c2abs, 'go', label='Coefficients through estimation')
xlabel('n \u279d', fontstyle='italic')
ylabel('Coefficients of f\u2082(x) = cos(cos(x)) \u279D', fontstyle='italic')
title("Q3. Log-log plot of Fourier coefficients of cos(cos(x)) vs n")
legend(loc='upper right')
grid(True)
show()


# The maximum absolute deviation is calculated for the estimates, for each of the functions:
expmaxdev=max([abs(expco[m]-c1[m]) for m in range(51)])
coscosmaxdev=max([abs(coscosco[m]-c2[m]) for m in range(51)])

print('Largest absolute deviation in exp(x) coefficients=',expmaxdev)
print('Largest absolute deviation in cos(cos(x)) coefficients=',coscosmaxdev)

#Function values are estimated from the function obtained from the Fourier Series
expest=matmul(A,c1)
coscosest=matmul(A,c2)

#Figures 1 and 2 are re-plotted, with the function values obtained using the estimated Fourier Coefficients marked in green circles:
epoints=eexp(x)
coscospoints=coscos(x)
fourierexp=eexp(x%(2*pi))
fouriercoscos=coscos(x%(2*pi))
#The above data points are re-calculated because we have redefined the x values after the previous calculation 
figure('Figure 1')
semilogy(x, epoints, label='True function')
semilogy(x, fourierexp, label='Expected function from Fourier Series')
semilogy(x, expest,  'go', label='Values through estimated coefficients')
xlabel('x \u279d', size=15, fontstyle='italic')
ylabel('e\u02E3 \u279D', size=15, fontstyle='italic')
title("Q7. Semilog(y) Plot of e\u02E3 vs x")
legend(loc='lower right')
grid(True)
show()
figure('Figure 2')
plot(x,coscospoints, label='True function')
plot(x, fouriercoscos, label='Expected function from Fourier Series')
plot(x, coscosest, 'go', label='Values through estimated coefficients')
xlabel('x \u279d', size=15, fontstyle='italic')
ylabel('cos(cos(x)) \u279D', size=15, fontstyle='italic')
title("Q7. Plot of cos(cos(x)) vs x")
legend(loc='upper right')
grid(True)
show()



