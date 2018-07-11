import numpy as np 
import pandas as pd
import scipy.stats as sps
import math
import matplotlib.pyplot as plt 
from matplotlib import style
from matplotlib.ticker import MultipleLocator 
import datetime



# read data files and get the info 
marketData = pd.read_csv('marketdata.csv')
IVdf = pd.read_csv('instruments.csv')
strike = IVdf.Strike.tolist()
optionS = IVdf.Symbol.tolist()
strike.pop()
optionS.pop()
IVdf_P = IVdf.loc[IVdf['OptionType'] == 'P']
strike_P = IVdf_P.Strike.tolist()
strike_P.sort()
IVdf_C = IVdf.loc[IVdf['OptionType'] == 'C']
strike_C = IVdf_C.Strike.tolist()
strike_C.sort()


#global variables 
r = 0.04
q = 0.2
T = 0.02191781

#form a dataframe contains the data of underlying 
marketData['Price'] = (marketData['Bid1'] + marketData['Ask1'])/2
Stock = marketData.loc[marketData['Symbol']==510050,['LocalTime']]
stock_price = marketData.loc[marketData['Symbol']==510050].Price
stock_p = stock_price.tolist()
Stock['price'] = stock_p

# define a function to calculate option value 
def bs_pricing(optionType, s, k, t, T, sigma, r, q = 0):
	optionType = 'Call' if optionType == 'C' else 'Put'
	optionSign = 1.0 if (optionType == 'Call') else -1.0
	s = float(s)
	k = float(k)
	t = float(t)
	T = float(T)
	sigma = float(sigma)
	r = float(r)

	d1 = (np.log(s/k)+(r-q)*(T-t))/(sigma*np.sqrt(T-t)) + 0.5*sigma*np.sqrt(T-t)
	d2 = (np.log(s/k)+(r-q)*(T-t))/(sigma*np.sqrt(T-t)) - 0.5*sigma*np.sqrt(T-t)

	optionValue = optionSign*s*np.exp(-q*(T-t))*sps.norm.cdf(optionSign*d1)-optionSign*k*np.exp(-r*(T-t))*sps.norm.cdf(optionSign*d2)
	return optionValue

# define a function to calculate vage
def Vega(s, k, tau, sigma, r, q = 0):
	s = float(s)
	k = float(k)
	tau = float(tau)
	sigma = float(sigma)
	r = float(r)

	d1 = (np.log(s/k)+(r-q)*(tau))/(sigma*np.sqrt(tau)) + 0.5*sigma*np.sqrt(tau)
	vega = s*np.sqrt(tau)*np.exp(-0.5*d1*d1)/np.sqrt(2*math.pi)
	return vega

# define a function to get market price of options at a given time point from our dataset 
def optionPrice(optionType, priceType, k, time_point):
	d1 = datetime.datetime.strptime(time_point, '%Y-%b-%d %H:%M:%S.%f') 
	#find the option 
	x = IVdf.loc[IVdf['OptionType'] == optionType].loc[IVdf['Strike'] == k].Symbol.tolist()
	y = x[0]
	#form a dataframe contains all the information of that option
	z = marketData.loc[marketData['Symbol']==y] 
	d=[]
	d0=[]
	#form a list contains all the time point for that option
	for index,row in z.iterrows(): 
		t = z.loc[index,['LocalTime']].to_string().split('    ')
		d.append(t[1]) #a list of time point for that option, format: string

	#form a list contains the gap between each time point and time point passed in
	for i in range(len(d)): 
		d2 = datetime.datetime.strptime(d[i], '%Y-%b-%d %H:%M:%S.%f')
		# calculate the gap 
		d_gap = d2 - d1 
		#transform the data type to flaot 
		d_gap = d_gap.total_seconds()
		# a list contains all the time gaps for that option
		d0.append(d_gap)  
	# append the list to z as a new column 
	z.insert(7, 'time_gap', d0)
	# form a dataframe contains all information of that option whose time gap is minus
	zz = z.loc[z.time_gap < 0] 
	# if there's no prices posted before
	if zz.empty:
		optionPrice = -9999
	else:
		idx = zz.time_gap.idxmax()
		xx = zz.loc[idx,[priceType]].tolist()
		optionPrice = xx[0]

	return optionPrice

# define a function to get the underlying price at a given time point
def stockPrice(time_point):
	d1 = datetime.datetime.strptime(time_point, '%Y-%b-%d %H:%M:%S.%f') 
	stock_time = Stock.LocalTime.tolist()
	d0 = []
	#form a list contains the gap between each time point and time point passed in
	for i in range(len(stock_time)): 
		d2 = datetime.datetime.strptime(stock_time[i], '%Y-%b-%d %H:%M:%S.%f')
		# calculate the gap 
		d_gap = d2 - d1 
		#transform the data type to flaot 
		d_gap = d_gap.total_seconds()
		# a list contains all the time gaps for that option
		d0.append(d_gap)  
	# append the list to z as a new column 
	Stock['time_gap'] = d0 
	# form a dataframe contains all information of that option whose time gap is minus
	Stock_a = Stock.loc[Stock.time_gap < 0] 
	# if there's no prices posted before
	if Stock_a.empty:
		stockPrice = -9999
	else:
		idx = Stock_a.time_gap.idxmax()
		xx = Stock_a.loc[idx,['price']].tolist()
		stockPrice = xx[0]

	return stockPrice

# define a function to calculate first guess of sigma 
def sigmahat(s, k, tau,r, q = 0): 
	sigmahat = np.sqrt(2*abs((np.log(s/k)+(r-q)*tau)/tau))
	return sigmahat

# define a function to implement newton method 
def sigma(optionType, priceType, k, tau, r, q, time_point):
	tol = 1e-8
	nmax = 500
	n = 1
	sigmadiff = 1.0

	s = stockPrice(time_point)
	option_price = optionPrice(optionType, priceType, k, time_point)
	if (s == -9999 or option_price == -9999):
		sigma = 'NaN'
	else:
		sigma = sigmahat(s, k, tau, r, q)
		while (sigmadiff >= tol  and  n < nmax):
			C = bs_pricing(optionType, s, k, 0, tau, sigma, r, q)
			vega = Vega(s, k, tau, sigma, r, q)
			increment = (C-option_price)/vega
			sigma = sigma-increment
			n = n+1
			sigmadiff = abs(increment)
	
	return sigma

# generate 31.csv, 32.csv, 33.csv
BidVolP1 = []
AskVolP1 = []
BidVolC1 = []
AskVolC1 = []	
df31 = pd.DataFrame()
df32 = pd.DataFrame()
df33 = pd.DataFrame()

for i in range(len(strike_P)):
	AVolP = sigma('P','Ask1',strike_P[i], 0.02191781,0.04, 0.2, '2016-Feb-16 09:31:00.00')
	AskVolP1.append(AVolP)

for i in range(len(strike_C)):
	AVolC = sigma('C','Ask1',strike_C[i],0.02191781,0.04, 0.2, '2016-Feb-16 09:31:00.00')
	AskVolC1.append(AVolC)

for i in range(len(strike_P)):
	BVolP = sigma('P','Bid1',strike_P[i],0.02191781,0.04, 0.2, '2016-Feb-16 09:31:00.00')
	BidVolP1.append(BVolP)

for i in range(len(strike_C)):
	BVolC = sigma('C','Bid1',strike_C[i],0.02191781,0.04, 0.2, '2016-Feb-16 09:31:00.00')
	BidVolC1.append(BVolC)

df31.insert(0, 'Strike', strike_P)
df31.insert(1,'BidVolP',BidVolP1)
df31.insert(2,'AskVolP', AskVolP1)
df31.insert(3,'BidVolC', BidVolC1)
df31.insert(4,'AskVolC', AskVolC1)
print('31.csv')
print(df31)
df31.to_csv('31.csv')

BidVolP2 = []
AskVolP2 = []
BidVolC2 = []
AskVolC2 = []	
for i in range(len(strike_P)):
	AVolP = sigma('P','Ask1',strike_P[i],0.02191781,0.04, 0.2, '2016-Feb-16 09:32:00.00')
	AskVolP2.append(AVolP)

for i in range(len(strike_C)):
	AVolC = sigma('C','Ask1',strike_C[i],0.02191781,0.04, 0.2, '2016-Feb-16 09:32:00.00')
	AskVolC2.append(AVolC)


for i in range(len(strike_P)):
	BVolP = sigma('P','Bid1',strike_P[i],0.02191781,0.04, 0.2, '2016-Feb-16 09:32:00.00')
	BidVolP2.append(BVolP)


for i in range(len(strike_C)):
	BVolC = sigma('C','Bid1',strike_C[i],0.02191781,0.04, 0.2, '2016-Feb-16 09:32:00.00')
	BidVolC2.append(BVolC)


df32.insert(0, 'Strike', strike_P)
df32.insert(1,'BidVolP',BidVolP2)
df32.insert(2,'AskVolP', AskVolP2)
df32.insert(3,'BidVolC', BidVolC2)
df32.insert(4,'AskVolC', AskVolC2)
print('32.csv')
print(df32)
df32.to_csv('32.csv')

BidVolP3 = []
AskVolP3 = []
BidVolC3 = []
AskVolC3 = []	

for i in range(len(strike_P)):
	AVolP = sigma('P','Ask1',strike_P[i],0.02191781,0.04, 0.2, '2016-Feb-16 09:33:00.00')
	AskVolP3.append(AVolP)

for i in range(len(strike_C)):
	AVolC = sigma('C','Ask1',strike_C[i],0.02191781,0.04, 0.2, '2016-Feb-16 09:33:00.00')
	AskVolC3.append(AVolC)

for i in range(len(strike_P)):
	BVolP = sigma('P','Bid1',strike_P[i],0.02191781,0.04, 0.2, '2016-Feb-16 09:33:00.00')
	BidVolP3.append(BVolP)

for i in range(len(strike_C)):
	BVolC = sigma('C','Bid1',strike_C[i],0.02191781,0.04, 0.2, '2016-Feb-16 09:33:00.00')
	BidVolC3.append(BVolC)
df33.insert(0, 'Strike', strike_P)
df33.insert(1,'BidVolP',BidVolP3)
df33.insert(2,'AskVolP', AskVolP3)
df33.insert(3,'BidVolC', BidVolC3)
df33.insert(4,'AskVolC', AskVolC3)
print('33.csv')
print(df33)
df33.to_csv('33.csv')

# generate plots
time0 = ['2016-Feb-16 09:31:00.00', '2016-Feb-16 09:32:00.00', '2016-Feb-16 09:33:00.00']
BidVolP = [BidVolP1, BidVolP2, BidVolP3]
AskVolC = [AskVolC1, AskVolC2, AskVolC3]
BidVolC = [BidVolC1, BidVolC2, BidVolC3]
AskVolP = [AskVolP1, AskVolP2, AskVolP3]
for i in range(3):
	x = strike_P
	y1 = BidVolP[i]
	y2 = AskVolP[i]
	y3 = BidVolC[i]
	y4 = AskVolC[i]
	plt.plot(x, y1, label = 'Put Vol for Bid', linewidth = 3, color = 'r', marker = 'o')
	plt.plot(x, y2, label = 'Put Vol for Ask', linewidth = 3, color = 'b', marker = 'x')
	plt.plot(x, y3, label = 'Call Vol for Bid', linewidth = 3, color = 'g', marker = '*')
	plt.plot(x, y4, label = 'Call Vol for Ask', linewidth = 3, color = 'y', marker = '^')
	plt.xlabel('Strike')
	plt.ylabel('Volatility')
	plt.title(time0[i])
	plt.legend(loc = 'upper left')
	plt.show()






