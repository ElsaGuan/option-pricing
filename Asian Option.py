import numpy as np
import math
import scipy.stats as sps

def geo_Asian(s0, k, optionType, n, tau, r, sigma,cv):
	sigmahat = sigma*np.sqrt(((n+1)*(2*n+1))/(6*n*n))
	muhat = (r-0.5*sigma*sigma)*(n+1)/(2*n) + 0.5*sigmahat*sigmahat

	d1 = (log(s0/k) + (muhat+0.5*sigmahat*sigmahat)*tau)/(sigmahat*np.sqrt(tau))
	d2 = d1 - sigmahat*np.sqrt(tau)

	if optionType == 'c':
		geo_v = math.exp(-r*tau)*(s0*math.exp(muhat*tau)*sps.norm.cdf(d1)-k*sps.norm.cdf(d2))
	elif optionType == 'p':
		geo_v = math.exp(-r*tau)*(-s0*math.exp(muhat*tau)*sps.norm.cdf(-d1)+k*sps.norm.cdf(-d2))

	return geo_v

def arith_Asian(s0,k,optionType,m,n,tau,r,sigma,cv):
	dt = tau/n
	# growthFactor = math.exp((r-0.5*sigma*sigma)*dt+sigma*np.sqrt(dt)*np.random.normal(0,1))
	# Spath = [s*growthFactor]
	Spath = []
	arith_v = []
	geo_v = []

	for i in range(m):
		growthFactor = math.exp((r-0.5*sigma*sigma)*dt+sigma*np.sqrt(dt)*np.random.normal(0,1))
		Spath[0] = [s*growthFactor]
		for j in range(1,n):
			sj = Spath[j-1]*growthFactor
			Spath.append(sj)
		Spath = np.array(Spath)
		arith_mean = Spath.mean()
		geo_mean = sps.gmean(Spath)

		if optionType == 'c':
			arith_payoff = math.exp(-r*tau)*max(arith_mean-k,0)
			arith_v.append(arith_payoff)
			geo_payoff = math.exp(-r*tau)*max(geo_mean - k, 0)
			geo_v.append(geo_payoff)
		else:
			arith_payoff = math.exp(-r*tau)*max(k-arith_mean,0)
			arith_v.append(arith_payoff)
			geo_payoff = math.exp(-r*tau)*max(k-geo_mean, 0)
			geo_v.append(geo_payoff)
	geo_v = np.array(geo_v)
	arith_v = np.array(arith_v)

	if cv == 1:
		covXY = (geo_payoff*arith_payoff).mean() - geo_payoff.mean()*arith_payoff.mean()
		var_geo = geo_payoff.std()
		theta = covXY*covXY/var_geo
		arithhat = []
		for i in range(arith_payoff):
			x = arith_payoff[i] + theta*(geo_Asian(s0, k, optionType, n, tau, r, sigma,1) - geo_payoff[i])
			arithhat.append(x)
		arithhat = np.array(arithhat)	
		arith_cv_mean = arithhat.mean()
		arith_cv_std = arithhat.std()
		arith_cv_value = [arith_cv_mean - 1.96*arith_cv_std/np.sqrt(m), arith_cv_mean + 1.96*arith_cv_std/np.sqrt(m)]
		return arith_cv_value
	else:
		arithMean = arith_v.mean()
		arithStd = arith_v.std()
		arith_value = [arithMean - 1.96*arithStd/np.sqrt(m), arithMean + 1.96*arithStd/np.sqrt(m)]
		return arith_value