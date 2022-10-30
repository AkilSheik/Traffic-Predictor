import copy, math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Settings
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
np.set_printoptions(precision= 3)


#Parsing the data into a numpy array
data= pd.read_csv(r"C:\Users\akils\Downloads\imports-85.data")
data = data.apply(pd.to_numeric, errors = 'coerce').dropna(axis=1, how='all').dropna(axis = 0, how = 'any')
data = data.to_numpy(dtype = np.float64)

#The machine learning begins
X_train = data[:,2:-1]
Y_train = np.array(data[:, -1])
m,n  = X_train.shape
w_init = np.zeros(n)
b_init = 0
mu = np.mean(X_train, axis = 0)
sigma = np.std(X_train, axis = 0)
X_train = (X_train-mu)/sigma
def compute_cost(X, y, w, b): 
    """
    compute cost
    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters  
      b (scalar)       : model parameter
      
    Returns:
      cost (scalar): cost
    """
    m = X.shape[0]
    cost = 0.0
    for i in range(m):                                
        f_wb_i = np.dot(X[i], w) + b           #(n,)(n,) = scalar (see np.dot)
        cost = cost + (f_wb_i - y[i])**2       #scalar
    cost = cost / (2 * m)                      #scalar    
    return cost


def compute_gradient(X, y, w, b): 
    """
    Computes the gradient for linear regression 
    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters  
      b (scalar)       : model parameter
      
    Returns:
      dj_dw (ndarray (n,)): The gradient of the cost w.r.t. the parameters w. 
      dj_db (scalar):       The gradient of the cost w.r.t. the parameter b. 
    """
    m,n = X.shape           #(number of examples, number of features)
    
    #m is the number of examples
    #n is the number of features
    
    #intilizes the gradients
    dj_dw = np.zeros((n,))
    dj_db = 0.
    
    #For each training example

    for i in range(m):
        #Find its error
        err = (np.dot(X[i], w) + b) - y[i]
        
        #for every feature of the training example. Overall, forms the dj_dw gradients for the ith training example
        for j in range(n):
            #Add to each feature of itself the error * the training example's ith feature
            dj_dw[j] = dj_dw[j] + err * X[i, j]    
        
        dj_db = dj_db + err                        
    dj_dw = dj_dw / m                                
    dj_db = dj_db / m                                
        
    return dj_db, dj_dw



def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters): 
    """
    Performs batch gradient descent to learn theta. Updates theta by taking 
    num_iters gradient steps with learning rate alpha
    
    Args:
      X (ndarray (m,n))   : Data, m examples with n features
      y (ndarray (m,))    : target values
      w_in (ndarray (n,)) : initial model parameters  
      b_in (scalar)       : initial model parameter
      cost_function       : function to compute cost
      gradient_function   : function to compute the gradient
      alpha (float)       : Learning rate
      num_iters (int)     : number of iterations to run gradient descent
      
    Returns:
      w (ndarray (n,)) : Updated values of parameters 
      b (scalar)       : Updated value of parameter 
      """
    
    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    w = copy.deepcopy(w_in)  #avoid modifying global w within function
    b = b_in
    
    for i in range(num_iters):

        # Calculate the gradient and update the parameters
        dj_db,dj_dw = gradient_function(X, y, w, b)   ##None

        # Update Parameters using w, b, alpha and gradient
        w = w - alpha * dj_dw               ##None
        b = b - alpha * dj_db               ##None
      
        # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion 
            J_history.append( cost_function(X, y, w, b))

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]}, dj_db {dj_db}  ")
        
    return w, b, J_history #return final w,b and J history for graphing

w_final, b_final, J_hist = gradient_descent(X_train, Y_train, w_init, b_init,
                                                    compute_cost, compute_gradient, 
                                                    0.1, 100000)
print(w_final)
print(b_final)