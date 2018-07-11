import numpy as np
import math

def American_option(s0,k,optionType,n,tau,r,sigma):
	dt = tau/n
	u = math.exp(sigma*np.sqrt(dt))
	d = 1/u
	p = (math.exp(r*dt)-d)/(u-d)

	underlying_tree = np.zeros((n+1,n+1))
	underlying_tree[0,0] = s0
	for i in range(1,n+1):
		for j in range(i+1):
			underlying_tree[j,i] = s0*np.power(u,i-j)*np.power(d,j)


	if optionType == 'c':
		optionSign = 1
	else:
		optionSign = -1

	option_tree = np.zeros((n+1,n+1))
	option_tree[:,n] = [max(optionSign*(underlying_tree[i,n]-k),0) for i in range(n+1)]
	for i in range(n-1,-1,-1):
		for j in range(i+1):
			option_tree[j,i] = max((underlying_tree[j,i]-k)*optionSign, (option_tree[j+1,i+1]*(1-p)+option_tree[j,i+1]*p)*math.exp(-r*dt))
	return option_tree[0,0]