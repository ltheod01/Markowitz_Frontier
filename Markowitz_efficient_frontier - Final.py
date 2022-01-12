import pandas_datareader.data as web
import datetime
import pandas as pd
import yfinance as yf
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
    
start_date = datetime.datetime(2012,3,1)
end_date = datetime.datetime(2022,1,6)


tickers = ['INDA','VOO','VOOV', 'VOOG', 'VFH', 'VHT',  'USRT', 'VNQ']
rdata = pd.DataFrame()

sample_df = yf.download(tickers, start = start_date, end = end_date, group_by= "ticker",interval ='1mo')

for item in tickers:
    i = sample_df
    rdata[item] = sample_df[item]['Adj Close']
    
rdata.dropna(inplace = True)

returns = 1 - rdata/rdata.shift(1)

#Calculating the var-cov matrix and it's inverse
S = returns.cov()
Sinv = pd.DataFrame(np.linalg.pinv(S.values), S.columns, S.index)
Sinv.dot(S)

#Creating 2 matrixes with averages 
avgret = returns.mean()
avgret2 = avgret - min(avgret)
sigma = returns.std()

##################################################################
#Finding the weights of 2 optimal portfolios
z1 = np.dot(Sinv,(avgret))
z2 = np.dot(Sinv, avgret2)

#w1 and w2 are the weights of 2 efficient portfolios
w1 = z1/np.sum(z1)
w2 = z2/np.sum(z2)

#We now find the return and standard deviation of the 2 efficient portfolios
rp1 = np.dot(np.transpose(avgret), w1)
var1 = np.dot(np.transpose(w1),np.dot(S,w1))
sigma1 = var1**(1/2)

rp2 = np.dot(np.transpose(avgret), w2)
var2 = np.dot(np.transpose(w2),np.dot(S,w2))
sigma2 = var2**(1/2)

##Now we will calculate one optimal portfolio and apply the same process
##for different weights to find the efficient frontier
frontier_weights = np.arange(-2.0,2.5,0.05)
efficient_frontier = pd.DataFrame(columns = ('Sigma', 'Returns'))

covar = np.dot(np.transpose(w1),np.dot(S,w2))

for weight in frontier_weights:
    
    rpp = weight*rp1+(1-weight)*rp2
    sigmapp = ((weight**2)*var1+((1-weight)**2)*var2+2*weight*(1-weight)*covar)**0.5

    values = pd.DataFrame({'Sigma':[sigmapp], 'Returns':[rpp]})
    efficient_frontier = efficient_frontier.append(values, ignore_index=True)


print(efficient_frontier)

plt.scatter(efficient_frontier['Sigma'],efficient_frontier['Returns'])
plt.show()


