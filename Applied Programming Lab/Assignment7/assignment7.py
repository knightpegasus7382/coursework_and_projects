#########################################################
#					ASSIGNMENT 7						#
#########################################################
#	Author: Akash Reddy A								#
#	Roll No: EE17B001									#
#########################################################

import pylab as p
from sympy import *
import scipy.signal as sp
import numpy as np
import math as m

def lowpass(R1,R2,C1,C2,G,Vi):
	s=symbols('s')
	A=Matrix([[0,0,1,-1/G],[-1/(1+s*R2*C2),1,0,0],[0,-G,G,1],[-1/R1-1/R2-s*C1,1/R2,0,s*C1]])
	b=Matrix([0,0,0,-Vi/R1])
	V=A.inv()*b
	return (A,b,V)
	
def highpass(R1,R3,C1,C2,G,Vi):
	s=symbols('s')
	A=Matrix([[0,0,1,-1/G],[-(s*R3*C2)/(1+s*C2*R3),1,0,0],[0,-G,G,1],[-s*C1-s*C2-1/R1,s*C2,0,1/R1]])
	b=Matrix([0,0,0,-Vi*s*C1])
	V=A.inv()*b
	return (A,b,V)

def transfer(x):
	s=symbols('s')	
	Vo=simplify(x[3])
	vonumden=fraction(Vo)
	vonum=Poly(vonumden[0],s)
	voden=Poly(vonumden[1],s)
	vonumf=[float(i) for i in vonum.all_coeffs()]
	vodenf=[float(i) for i in voden.all_coeffs()]
	v0=sp.lti(vonumf, vodenf)
	return (v0,Vo)
	
def plotter(x,y,i,xlbl,ylbl,ilbl,title):
	p.grid(True)
	p.plot(x,i,'-y',label=ilbl)
	p.plot(x,y,'-k',label=ylbl)
	p.xlabel(xlbl, size=12)
	p.title(title)
	p.legend(loc='upper right')
	p.subplots_adjust(hspace=0.8)
	p.show()
	
s=symbols('s')
w=p.logspace(0,8,801)
ss=1j*w
A,b,V=lowpass(10000,10000,1e-9,1e-9,1.586,1)
v0,Vo=transfer(V)
hf=lambdify(s,Vo,'numpy')
v=hf(ss)
p.loglog(w,abs(v),lw=2,label='Impulse response')
A,b,V=lowpass(10000,10000,1e-9,1e-9,1.586,1/s)
v0,Vo=transfer(V)
hf=lambdify(s,Vo,'numpy')
v=hf(ss)
p.loglog(w,abs(v),lw=2,label='Step response')
p.xlabel('$\omega$\u279d', size=12)
p.ylabel('$|H(j\omega)|\u279d$', size=12)
p.title('Magnitude response of the step and impulse responses in the frequency domain for low-pass filter')
p.legend(loc='upper right')
p.grid(True)
p.show()
t,vo=sp.impulse(v0,None,np.linspace(0,0.005,501)) 
plotter(t,vo,np.heaviside(t,0),'$t\u279d$','$v_o(t)=s(t)$','$v_i(t)=u(t)$','Q1: Input step $u(t)$ and step response $s(t)$ vs $t$ for low-pass filter')

A,b,V=lowpass(10000,10000,1e-9,1e-9,1.586,1)
v0,Vo=transfer(V)
t=np.linspace(0,1e-2,1e+5)
vi=(np.sin(2000*m.pi*t)+np.cos(2*10**6*m.pi*t))*np.heaviside(t,0)
t,vo,svec=sp.lsim(v0,vi,t)
plotter(t,vo,vi,'$t\u279d$','$v_o(t)$','$v_i(t)$','Q2: $v_i(t)=(\sin(2000\pi t)+\cos(2 x 10^6\pi t))u(t)$ and $v_o(t)$ vs $t$ for low-pass filter')

A,b,V=highpass(10000,10000,1e-9,1e-9,1.586,1)
v0,Vo=transfer(V)
t=np.linspace(0,0.5,1e+5)
vi=np.sin(1000*t)*np.exp(-5*t)*np.heaviside(t,0)
t,vo,svec=sp.lsim(v0,vi,t)
plotter(t,vo,vi,'$t\u279d$','$v_o(t)$','$v_i(t)$','Q4: Damped $v_i(t)=e^{-5t}\sin (1000t)u(t)$ and $v_o(t)$ vs $t$ for high-pass filter')

A,b,V=highpass(10000,10000,1e-9,1e-9,1.586,1)
v0,Vo=transfer(V)
hf=lambdify(s,Vo,'numpy')
v=hf(ss)
p.loglog(w,abs(v),lw=2,label='Impulse response')
A,b,V=highpass(10000,10000,1e-9,1e-9,1.586,1/s)
v0,Vo=transfer(V)
hf=lambdify(s,Vo,'numpy')
v=hf(ss)
p.loglog(w,abs(v),lw=2,label='Step response')
p.xlabel('$\omega$\u279d', size=12)
p.ylabel('$|H(j\omega)|\u279d$', size=12)
p.title('Magnitude response of the step and impulse responses in the frequency domain for high-pass filter')
p.legend(loc='upper right')
p.grid(True)
p.show()
t,vo=sp.impulse(v0,None,np.linspace(0,0.005,501)) 
plotter(t,vo,np.heaviside(t,0),'$t\u279d$','$v_o(t)=s(t)$','$v_i(t)=u(t)$','Q5: Input step $u(t)$ and step response $s(t)$ vs $t$ for high-pass filter')
