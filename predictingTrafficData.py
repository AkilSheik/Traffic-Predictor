import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

#Settings
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
np.set_printoptions(precision= 3)


#Parsing the data into a numpy array
data= pd.read_csv(r"C:\Users\akils\Downloads\imports-85.data")
data = data.apply(pd.to_numeric, errors = 'coerce').dropna(axis=1, how='all').dropna(axis = 0, how = 'any')
data = data.to_numpy(dtype = np.float64)

#The machine learning begins
X_train = data[:, 2:-1]
print(X_train)
Y_train = np.array(data[:, -1])
m,n  = X_train.shape
w_init = np.zeros(n)
b_init = 0
mu = np.mean(X_train, axis = 0)
sigma = np.std(X_train, axis = 0)
X_train = (X_train-mu)/sigma


def prediction(X_train, w_param, b_param):
    return np.dot(X_train,w_param ) + b_param

def compute_cost(X_param, Y_param, w_param,b_param ):
  cost = 0
  for i in range(m):
    pred = np.dot(X_train[i, :], w_param) + b_param
    cost += (pred - Y_param[i])**2
  cost /= (2 * m)
  return cost

def compute_gradient(X_train, Y_train, w_param, b_param):
    dj_dw = np.zeros(n)
    dj_db = 0
    for i in range(m): 
        pred = np.dot(X_train[i, :], w_param) + b_param
        err = pred - Y_train[i]
        for j in range(n):
           dj_dw[j] += err * X_train[i,j]
        dj_db += err
    dj_dw /= m
    dj_db /= m

    return dj_dw, dj_db

#Calling the algorithim
def graphData(feature, w, b):
    X_features = ['0', '1', '2', '3', '4','5','6','7', '8', '9','10', '11','12']
    predicted = np.zeros(m)
    for i in range(m):
        predicted[i] = w[feature] * X_train[i][feature] + b
    plt.plot(X_train[:, feature], predicted, c = "b")
    plt.scatter(X_train[:, feature], Y_train, marker='x', c='r') 
    plt.title("Cost vs. w{feature}")
    plt.ylabel('Profit')
    plt.xlabel(f'w{feature}')
    plt.show()

def impliment_gradient_descent(X_train, Y_train, w_param, b_param, learning_rate, number_iters):
    costHistory = []
    
    for i in range(number_iters):
        dj_dw, dj_db = compute_gradient(X_train, Y_train, w_param, b_param)
        w_param -= learning_rate * dj_dw
        b_param -= learning_rate * dj_db
        if i % 100 == 0 :
            costHistory.append(compute_cost(X_train, Y_train, w_param, b_param))
            print(f"Iteration: {i} Cost: {costHistory[-1]}")
    return w_param, b_param
        

w,b = impliment_gradient_descent(X_train, Y_train, w_init, b_init, 0.1, 10000)
print(w)
print(b)
#for x in range(n):
#  graphData(x, w, b)
print(X_train[:, 0])
print(X_train[:, 1])
print(Y_train.astype(int))




#fig,ax=plt.subplots(1, 2, figsize=(12, 3), sharey=False)
#for i in range(len(ax)):
#    ax[i].scatter(X_train[:,i],Y_train)
#    ax[i].set_xlabel(X_features[i])
#    x1, y1 = [0, b], [1, w[i]+b]
#    ax[i].plot(x1, y1, marker = 'o')
#ax[0].set_

#print(w)
#print(b)





