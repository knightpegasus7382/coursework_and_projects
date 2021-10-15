#########################################################
#					ASSIGNMENT 6						#
#########################################################
#	Author: Akash Reddy A								#
#	Roll No: EE17B001									#
#########################################################

######################################### QUESTION 1 #############################################
import scipy.signal as sp
import numpy as np
from pylab import *
H=sp.lti([1],[1,0,2.25])
F=sp.lti([1,0.5],polyadd(polymul([1,0.5],[1,0.5]),[2.25]))
X=sp.lti(polymul(F.num,H.num), polymul(F.den,H.den))
t,x=sp.impulse(X,None,np.linspace(0,50,501))
plot(t,x)
xlabel('$t\u279d$', size=15)
ylabel('$x(t)\u279d$', size=15)
title('Q1: Plot of $x(t)$ vs $t$ for the system $\ddot{x} + 2.25x = f(t)$, where $f(t)$ is the input')
grid(True)
show()

######################################### QUESTION 2 #############################################
F=sp.lti([1,0.05],polyadd(polymul([1,0.05],[1,0.05]),[2.25]))
X=sp.lti(polymul(F.num,H.num), polymul(F.den,H.den))
t,x=sp.impulse(X,None,np.linspace(0,50,501))
plot(t,x)
xlabel('$t\u279d$', size=15)
ylabel('$x(t)\u279d$', size=15)
title('Q2: Plot of $x(t)$ vs $t$ for the system $\ddot{x} + 2.25x = f(t)$, where $f(t)$ decays slower')
grid(True)
show()

######################################### QUESTION 3 #############################################
freq=np.arange(1.4,1.61,0.05)
for i in freq:
	f=np.cos(i*t)*np.exp(-0.05*t)*np.heaviside(t, 0.5)
	t,x,svec=sp.lsim(H, f, t)
	plot(t,x, label='$\omega=$'+str(i))
xlabel('$t\u279d$', size=15)
ylabel('$x(t)\u279d$', size=15)
title('Q3: Plot of $x(t)$ vs $t$ for the system $\ddot{x} + 2.25x = f(t)$, where $f(t)$ varies in frequency')
legend(loc='upper right')
grid(True)
show()

######################################### QUESTION 4 #############################################
X=sp.lti([1,0,2,0],[1,0,3,0,0])
Y=sp.lti([2,0],[1,0,3,0,0])
t,x=sp.impulse(X,None,np.linspace(0,20,201))
t,y=sp.impulse(Y,None,np.linspace(0,20,201))
plot(t,x,label='$x(t)$')
plot(t,y,label='$y(t)$')
xlabel('$t\u279d$', size=15)
title('Q4: Plots of $x(t)$ and $y(t)$ vs $t$ for the coupled equations')
legend(loc='upper right')
grid(True)
show()

######################################### QUESTION 5 #############################################
H=sp.lti([10**12],[1,10**8,10**12])
w,S,phi=H.bode()
subplot(2,1,1)
grid(True)
semilogx(w,S)
title('Q5: Bode Plot of the steady state transfer function of the given circuit')
ylabel('$|H(s)| (dB)\u279d$', size=15)
xlabel('$\omega\u279d$', size=15)
subplot(2,1,2)
grid(True)
semilogx(w,phi)
ylabel('$\u2220H(s)\u279d$', size=15)
xlabel('$\omega\u279d$', size=15)
show()

######################################### QUESTION 6 #############################################
t=np.linspace(0,10**-2,50001)
vi=(np.cos(10**3*t)-np.cos(10**6*t))*np.heaviside(t, 0.5)
t,vo,svec=sp.lsim(H, vi, t)
f=figure(1)
subplot(2,1,1)
plot(t,vo)
xlabel('$t\u279d$', size=15)
ylabel('$v_o(t)\u279d$', size=15)
title('Q6:Plot of $v_o(t)$ vs $t$ for the given input $v_i(t)$ to the circuit (for $t\leq 10 ms$)')
grid(True)
subplot(2,1,2)
plot(t,vi)
xlabel('$t\u279d$', size=15)
ylabel('$v_i(t)\u279d$', size=15)
title('Q6:Plot of input $v_i(t)$ to the circuit vs $t$ (for $t\leq 10 ms$)')
grid(True)
subplots_adjust(hspace=0.8)
t=np.linspace(0,3*10**-5,3001)
vi=(np.cos(10**3*t)-np.cos(10**6*t))*np.heaviside(t, 0.5)
t,vo,svec=sp.lsim(H, vi, t)
g=figure(2)
subplot(2,1,1)
plot(t,vo)
xlabel('$t\u279d$', size=15)
ylabel('$v_o(t)\u279d$', size=15)
title('Q6:Plot of $v_o(t)$ vs $t$ for the given input $v_i(t)$ to the circuit (First 30 $\mu$s)')
grid(True)
subplot(2,1,2)
plot(t,vi)
xlabel('$t\u279d$', size=15)
ylabel('$v_i(t)\u279d$', size=15)
title('Q6:Plot of input $v_i(t)$ to the circuit vs $t$ (First 30 $\mu$s)')
grid(True)
subplots_adjust(hspace=0.8)
show()
