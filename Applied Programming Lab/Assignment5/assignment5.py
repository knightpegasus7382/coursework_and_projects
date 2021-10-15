#########################################################
#					ASSIGNMENT 5						#
#########################################################
#	Author: Akash Reddy A								#
#	Roll No: EE17B001									#
#########################################################

from pylab import *
import mpl_toolkits.mplot3d.axes3d as p3
import sys
import numpy as np
import scipy.linalg as la
try:
	Nx=int(sys.argv[1])
	Ny=int(sys.argv[2])
	radius=float(sys.argv[3])
	Niter=int(sys.argv[4])
	xaxis_N=np.array(list(range(Niter)))
	phi=zeros((Ny,Nx))
	x=np.arange(float(-(Nx-1)/2),float((Nx+1)/2),1)
	y=np.arange(float(-(Ny-1)/2),float((Ny+1)/2),1)
	X,Y=meshgrid(x,y)
	ii=where((X/Nx)**2+(Y/Ny)**2<=(radius*radius))
	phi[ii]=1.0
	cp=contour(X,Y,phi[-1::-1,:])
	clabel(cp, cp.levels[0:7], inline=True, fontsize=7)
	xlabel('$X\u279d$', size=15)
	ylabel('$Y\u279d$', size=15)
	title('The contour plot of the initial potentials')
	plot(ii[1]-(Nx-1)/2,ii[0]-(Ny-1)/2,'ro',label='V = 1')
	legend(loc='upper right')
	show()
	
	errors=zeros((Niter))
	for k in range(Niter):
		oldphi=phi.copy()
		phi[1:-1,1:-1]=0.25*(phi[1:-1,0:-2]+phi[1:-1,2:]+phi[0:-2, 1:-1]+phi[2:,1:-1])
		phi[1:-1,0]=phi[1:-1,1]
		phi[1:-1,-1]=phi[1:-1,-2]
		phi[0,:]=phi[1,:]
		phi[ii]=1.0
		errors[k]=(abs(phi-oldphi)).max()
	C1=np.log(errors)
	C2=-1*ones((len(errors)))
	M=c_[C1,C2]
	b=xaxis_N
	x=la.lstsq(M,b)[0]
	x[0]=1/x[0]
	x[1]=x[1]*x[0]
	fit1=np.exp(x[1]+x[0]*b)
	C1=np.log(errors[500:])
	C2=-1*ones((len(errors[500:])))
	M=c_[C1,C2]
	b2=xaxis_N[500:]
	x=la.lstsq(M,b2)[0]
	x[0]=1/x[0]
	x[1]=x[1]*x[0]
	fit2=np.exp(x[1]+x[0]*b)
	print("Error at the 600th iteration=", errors[599])
	print("Error at the 800th iteration=", errors[799])
	semilogy(xaxis_N, errors, '-', label='errors')
	semilogy(xaxis_N, fit1, '-', label='fit1')
	semilogy(xaxis_N, fit2, '-', label='fit2')
	semilogy(list(range(0,Niter,50)), errors[::50], 'go')
	xlabel('Number of iterations\u279d', fontstyle='italic', size=12)
	ylabel('Magnitude of error\u279d', fontstyle='italic', size=12)
	title('Evolution of error with number of iterations (semilog(y))')
	legend(loc='upper right')
	show()
	
	loglog(xaxis_N, errors, '-', label='errors')
	loglog(xaxis_N, fit1, '-', label='fit1')
	loglog(xaxis_N, fit2, '-', label='fit2')
	loglog(list(range(0,Niter,50)), errors[::50], 'go')
	xlabel('Number of iterations\u279d', fontstyle='italic', size=12)
	ylabel('Magnitude of error\u279d', fontstyle='italic', size=12)
	title('Evolution of error with number of iterations (loglog)')
	legend(loc='upper right')
	show()
	
	fig1=figure(4)
	ax=p3.Axes3D(fig1)
	title('The 3-D surface plot of the potential')
	surf = ax.plot_surface(X,Y, phi[-1::-1,:], rstride=1, cstride=1, cmap=cm.jet)
	ax.set_xlabel('$X$', size=20)
	ax.set_ylabel('$Y$', size=20)
	ax.set_zlabel('$\phi$', size=20)
	show()
	
	cp=contour(X,Y,phi[-1::-1,:])
	clabel(cp, cp.levels[0:7], inline=True, fontsize=7)
	plot(ii[1]-(Nx-1)/2,ii[0]-(Ny-1)/2,'ro',label='V = 1')
	xlabel('$X\u279d$', size=15)
	ylabel('$Y\u279d$', size=15)
	title('The contour plot of the updated potentials')
	legend(loc='upper right')
	show()
	
	Jx=zeros((Ny,Nx))
	Jy=zeros((Ny,Nx))
	Jx[1:-1,1:-1]=0.5*(phi[1:-1,0:-2]-phi[1:-1,2:])
	Jy[1:-1,1:-1]=0.5*(phi[2:,1:-1]-phi[0:-2:,1:-1])
	quiver(X,Y,Jx[-1::-1,:],Jy[-1::-1,:], scale=5)
	plot(ii[1]-(Nx-1)/2,ii[0]-(Ny-1)/2,'ro', label='V = 1')
	xlabel('$X\u279d$', size=15)
	ylabel('$Y\u279d$', size=15)
	legend(loc='upper right')
	title('The vector plot of the current flow')
	show()
	
	#HEAT
	Q=(Jx**2+Jy**2)
	Q[ii]=0
	cp=contourf(X,Y,Q[-1::-1,:],cmap=cm.hot)
	colorbar()
	plot(ii[1]-(Nx-1)/2,ii[0]-(Ny-1)/2,'ro',label='Electrode')
	xlabel('$X\u279d$', size=15)
	ylabel('$Y\u279d$', size=15)
	title('The contour plot of the heat generated')
	legend(loc='upper right')
	show()
	
	#TEMPERATURE
	T=zeros((Ny,Nx))
	T[ii]=300
	T[-1]=300
	deltax=1/Nx
	deltay=1/Ny
	for k in range(Niter):
		T[1:-1,1:-1]=0.25*(T[1:-1,0:-2]+T[1:-1,2:]+T[0:-2, 1:-1]+T[2:,1:-1]+(Q[1:-1,1:-1]*(deltax**2)*(deltay**2))/0.01	)
		T[1:-1,0]=T[1:-1,1]
		T[1:-1,-1]=T[1:-1,-2]
		T[0,:]=T[1,:]
		T[ii]=300
		T[-1]=300
	cp=contourf(X,Y,T[-1::-1,:], cmap=cm.jet)
	colorbar(format='%.6f')
	plot(ii[1]-(Nx-1)/2,ii[0]-(Ny-1)/2,'ro',label='Electrode')
	xlabel('$X\u279d$', size=15)
	ylabel('$Y\u279d$', size=15)
	title('The contour plot of the temperature')
	legend(loc='upper right')
	show()	
except Exception:
	print("Invalid user arguments: please enter valid values for Nx, Ny, radius, Niter in that order!")
	




