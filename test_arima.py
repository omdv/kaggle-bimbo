# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 14:18:43 2016

@author: OMedvedev
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

days = np.array([3,4,5,6,7,8,9])
vals = np.array([50.,45.,24.,30.,34.,18.,10.])
df = pd.DataFrame(np.vstack((days,vals)).T,\
        columns=['week','vals']).\
        set_index('week')

# Median
mean_val = df.vals.mean()

# Linear Fit
result = []
N = df.shape[0]
X = df.index.values
Y = df.vals.values
Mx = np.mean(X)
My = np.mean(Y)
b = (np.sum(X*Y.reshape(1,-1))-N*Mx*My)/(np.sum(X**2)-N*Mx*Mx)
a = My - b*Mx
val10 = (a+b*10)
val11 = (a+b*11)
if val10 < 0:
    val10 = 0
if val11 < 0:
    val11 = 0
    
#modelARIMA = ARIMA()
#modelARIMA.fit()

fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(df.vals.values, lags=6, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(df.vals, lags=6, ax=ax2)

arma20 = sm.tsa.ARMA(df.vals.values, (2,0), freq='W').fit()
arma_p10 = arma20.predict(5,8)

fig, ax = plt.subplots(figsize=(12, 8))
ax = df.plot(ax=ax)
fig = arma20.plot_predict(10,11,ax=ax,plot_insample=False)