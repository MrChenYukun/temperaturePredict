import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

def generateDate():
    np.random.seed(0)
    x_data = np.linspace(0,30,30)
    y_data = 2* np.sin(2*np.pi*x_data) + np.random.random(30)

    print(y_data)
    return x_data, y_data

def guessModel(x,a,b):
    return a*np.sin(b*np.pi*x)

def fittingData(x_data,y_data,testFunc):
    print("----")
    params, params_covariance = optimize.curve_fit(testFunc,x_data,y_data,p0=[2,2])
    return params

x_data, y_data = generateDate()

plt.figure(figsize=(6,4))
plt.scatter(x_data,y_data)
params = fittingData(x_data, y_data, guessModel)
print(params)

plt.figure(figsize=(6,4))
plt.scatter(x_data,y_data,label='Date')
plt.plot(x_data, guessModel(x_data,params[0],params[1]),label='Fitted function')

plt.legend(loc='best')
plt.show()