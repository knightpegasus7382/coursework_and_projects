#############################################
#			    ASSIGNMENT 3				#
#############################################
# Author: Akash Reddy A						#
# Roll Number: EE17B001						
#############################################

# Upon running the program, all graphs are displayed in order of questions, one after another #

from pylab import *
from numpy import *
from scipy.special import *
from scipy.linalg import *
def truncate(n,m):
	return int(n*(10**m))/(10**m)
################################ QUESTION 2 ##################################
data=loadtxt("fitting.dat",dtype=float,comments="#")
time=[m[0] for m in data]
data=[[m[z] for m in data] for z in range(1,10)]
k=9
scl=logspace(-1, -3, k)
################################ QUESTION 4 ##################################
def g(t, A, B):
	return A*jn(2,t)+B*t
################################ QUESTION 3 ##################################
SUB=str.maketrans("0123456789","\u2080\u2081\u2082\u2083\u2084\u2085\u2086\u2087\u2088\u2089")
for x in range(len(data)):
	plot(time, data[x], label= "\u03C3"+str(x+1).translate(SUB)+"=" +str(truncate(scl[x],4)))
plot(time, g(array(time), 1.05,-0.105), '-k', label = 'True Value')
legend(loc='upper right')
xlabel('t \u279d', size=15, fontstyle='italic')
ylabel('f(t) + noise \u279d', size=15, fontstyle='italic')
title('Q4: Data to be fitted to theory')
grid(True)
show()
################################ QUESTION 5 ##################################
errorbar(time[::5], data[0][::5], scl[0], fmt='ro', label = 'errorbar')
plot(time, g(array(time), 1.05,-0.105), '-k', label = 'f(t)')
legend(loc='upper right')
xlabel('t \u279d', size=15, fontstyle='italic')
title('Q5: Data points for \u03C3=0.10 along with exact function f(t)')
grid(True)
show()
################################ QUESTION 6 ##################################
J=array([jn(2, m) for m in time])
T=array([m for m in time])
M=c_[J,T]
################################ QUESTION 7,8 ##################################
Alist=arange(0,2.1,0.1)
Blist=arange(-0.20,0.01,0.01)
A,B=meshgrid(Alist, Blist)
epsilon=0
for k in range(len(time)):
	epsilon+=1/101*((data[0][k]-g(time[k], A,B))**2)
epsilon=transpose(epsilon)
plot(lstsq(M, data[0])[0][0], lstsq(M, data[0])[0][1], 'ro', label="Exact location")
cp=contour(A,B, epsilon, 20)
clabel(cp, cp.levels[0:5], inline=True, fontsize=10)
xlabel('A \u279d', size=15, fontstyle='italic')
ylabel('B \u279d', size=15, fontstyle='italic')
title('Q8: Contour plot of \u03B5\u1D62\u2C7C')
show()
################################ QUESTION 9,10 #################################
G=[array(m) for m in data]
x=[lstsq(M, Gx) for Gx in G]
A_est=array([i[0][0] for i in x])
B_est=array([i[0][1] for i in x])
A_act=array([1.05 for i in range(len(x))])
B_act=array([-0.105 for i in range(len(x))])
plot(scl, abs(A_act-A_est), '--ro', label = 'Aerr', linewidth=0.5)
plot(scl, abs(B_act-B_est), '--go', label = 'Berr', linewidth=0.5)
legend(loc='upper left')
title('Q10: Variation of error with noise')
xlabel('Noise standard deviation \u279d', size=12, fontstyle='italic')
ylabel('MS error \u279d', size=12, fontstyle='italic')
grid(True)
show()
################################ QUESTION 11 ##################################
loglog(scl, abs(A_act-A_est), '--ro', label = 'Aerr', linewidth=0.5)
loglog(scl, abs(B_act-B_est), '--go', label = 'Berr', linewidth=0.5)
legend(loc='upper left')
title('Q11: Variation of error with noise (log-log plot)')
xlabel('\u03C3\u2099 \u279d', size=12, fontstyle='italic')
ylabel('MS error \u279d', size=12, fontstyle='italic')
grid(True)
show()
