#logistick
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import optimize

def generateDate():
    np.random.seed(0)

    data = pd.read_csv("soilTemp.csv")
    year_data = data.iloc[:,1]
    day_data = data.iloc[:,2]
    temp_data = data.iloc[:,3]
    y_data = []

    for i in range(len(year_data)):
        if(temp_data[i] < 1000):
            y_data.append(temp_data[i])

    x_data = np.linspace(0,len(y_data),len(y_data))
    return x_data, y_data

def guessModel_sin(x,a,b,c,d):
    return a*np.sin(np.pi/5.5*b*x + c) + d

def guessModel(x,a,b,c,d,e,f,g,h,i):
    x = x/5.53
    return a + (b* np.cos(1*x) + c* np.sin(1*x)) + (d* np.cos(2*x) + e* np.sin(2*x)) + (f* np.cos(3*x) + g* np.sin(3*x)) + (h* np.cos(4*x) + i* np.sin(4*x))

def fittingData(x_data,y_data,testFunc):
    print("----")
    params, params_covariance = optimize.curve_fit(testFunc,x_data,y_data)
    return params

x_data, y_data = generateDate()

plt.figure(figsize=(30,4))
plt.scatter(x_data,y_data)
params_sin = fittingData(x_data, y_data, guessModel_sin)
params = fittingData(x_data, y_data, guessModel)
print(params)

from sklearn.ensemble import RandomForestRegressor

plt.figure(figsize=(30,4))

rfr = RandomForestRegressor(n_estimators=1000, random_state=0)
rfr.fit(x_data.reshape(-1,1), y_data)
predictions = rfr.predict(x_data.reshape(-1,1))

plt.scatter(x_data,y_data,label='Date')
plt.plot(x_data, predictions, 'r', label='rfr')
plt.plot(x_data, guessModel_sin(x_data,params_sin[0],params_sin[1],params_sin[2],params_sin[3]),label='sin')
plt.plot(x_data, guessModel(x_data,params[0],params[1],params[2],params[3],params[4],params[5],params[6],params[7],params[8]),label='FFT')
#plt.plot(x_data, guessModel(x_data,params[0],params[1]),label='Fitted function')


plt.legend(loc='best')
plt.show()