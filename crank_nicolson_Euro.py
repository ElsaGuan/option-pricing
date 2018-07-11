import numpy as np 
import math

def Crank_Nicolson_Euro(optiontype, s,k,r,q,T,sigma):
	n = 252*T
	m = 100
	Smax = 2*k
	dS = Smax/m
	dt = T/n
	dz = sigma*sqrt(3*dt)

	if optiontype == 'c':
		Ot = 1
	else:
		Ot = -1

    #calculate matrices A for explicit and implicit scheme
	a1 = [(1/(1+r*dt))*((-0.5*dt/dz)*(r-0.5*sigma*sigma)+sigma*sigma*dt*0.5/(dz*dz)) for i in range(1,m)]
	b1 = [(1/(1+r*dt))*(1-sigma*sigma*dt/(dz*dz)) for i in range(1,m)]
	c1 = [(1/(1+r*dt))*((0.5*dt/dz)*(r-0.5*sigma*sigma)+sigma*sigma*dt*0.5/(dz*dz)) for i in range(1,m)]

	a2 = [(0.5*dt/dz)*(r-0.5*sigma*sigma)- sigma*sigma*dt*0.5/(dz*dz) for i in range(1,m)]
	b2 = [1+sigma*sigma*dt*0.5/(dz*dz)+r*dt for i in range(1,m)]
	c2 = [-(0.5*dt/dz)*(r-0.5*sigma*sigma)- sigma*sigma*dt*0.5/(dz*dz) for i in range(1,m)]

	x=np.array
	A1=np.diag(x(b1))+np.diag(x(a1[1:m-1]),k=-1)+np.diag(x(c1[0:m-2]),k=1)
	print(A1)
	A2=np.diag(x(b2))+np.diag(x(a2[1:m-1]),k=-1)+np.diag(x(c2[0:m-2]),k=1)
    
    #generate option value grid 
	f = np.zeros((m+1,n+1))
	f[0,:]=0
	f[m,:]=[Ot*(Smax - k*math.exp(-r*(n-j)*dt)) for j in range(n+1)]
	f[:,n]=[np.maximum(Ot*(i*dS - k),0) for i in range(m+1)]
	f=np.matrix(np.array(f))


    #calculate option value 
	for j in range(n-1, -1, -1):
		z1 = np.zeros((m-1,1))
		z1[0] = (0.5*(sigma**2)*dt - 0.5*r*dt)*f[0,j+1]
		z1[m-2] = (0.5*((m-1)**2)*(sigma**2)*dt+0.5*(m-1)*r*dt)*f[m,j+1]

		z2 = np.zeros((m-1,1))
		z2[0] = (0.5*(sigma**2)*dt - 0.5*r*dt)*f[0,j]
		z2[m-2] = (0.5*((m-1)**2)*(sigma**2)*dt+0.5*(m-1)*r*dt)*f[m,j]

		z = (z1 + z2)*0.5

		I = np.eye(m-1)
		A1_new = (A1 + I)*0.5
		A2_new = (A2 + I)*0.5
		x = A1_new.dot(f[1:m,j+1])
		y = np.zeros((m-1,1))
		for i in range(m-1):
			y[i,0] = x[i] + z[i,0]
		f[1:m,j] = (np.linalg.inv(A2_new)).dot(y)
		
	

	stock = []
	for i in range(m):
		stock.append(i * dS)
    #if spot price of underlying exists in the matrix 
	if s in stock:
		return f[int(s/dS),0]
	#if spot price of underlying doesnt exsit in the matrix, use interpolation to get option value 
	else:
		s1_list = []
		s2_list = []
		s_list = [stock[i] - s for i in range(len(stock))]
		for i in s_list:
			if i > 0:
				s1_list.append(i)
			else:
				s2_list.append(abs(i))
		s1 = min(s1_list) + s
		s2 = -1*min(s2_list) + s
		i1 = int(s1/dS)
		i2 = int(s2/dS)
		v = f[i2,0]*((f[i1,0]/f[i2,0])**((s-s2)/(s1-s2)))
		return v



    	 





