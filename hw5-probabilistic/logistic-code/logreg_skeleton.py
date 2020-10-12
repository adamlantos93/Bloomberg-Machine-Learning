import pandas as pd
import logging
import numpy as np
from scipy.optimize import minimize
import statistics
import matplotlib.pyplot as plt




def feature_normalization(train, test):
    max_feat=[]
    shift=[]
    for i in range(len(train[0,:])):
        shift.append(statistics.mean(train[:,i]))
        max_feat.append(abs((train[:,i]-shift[i])).max())
        for j in range(len(train[:,i])):
            train[j,i]=(train[j,i]-shift[i])/max_feat[i]
    max_feat=[]
    shift=[]
    for i in range(len(test[0,:])):
        shift.append(statistics.mean(test[:,i]))
        max_feat.append(abs((test[:,i]-shift[i])).max())
        for j in range(len(test[:,i])):
            test[j,i]=(test[j,i]-shift[i])/max_feat[i]
    return train, test
	


def f_objective(theta, X, y, l2_param=1):
    '''
    Args:
        theta: 1D numpy array of size num_features
        X: 2D numpy array of size (num_instances, num_features)
        y: 1D numpy array of size num_instances
        l2_param: regularization parameter

    Returns:
        objective: scalar value of objective function
    '''
    R=0
    for i in range(y.size):
        s=y[i]*np.dot(theta,X[i,:])
        R=R-s/2+np.logaddexp(-s/2,s/2)  #I am adding log(1+exp(-s)) (to avoid overflow)
    J=R/y.size+l2_param*np.dot(theta,theta) 
    return J
    
    
def fit_logistic_reg(X, y, objective_function, l2_param=1):
    '''
    Args:
        X: 2D numpy array of size (num_instances, num_features)
        y: 1D numpy array of size num_instances
        objective_function: function returning the value of the objective
        l2_param: regularization parameter
        
    Returns:
        optimal_theta: 1D numpy array of size num_features
    '''
    w_0=np.ones(X.shape[1])  #not much changes if we initialize from zeros
    w=minimize(objective_function, w_0, args=(X, y, l2_param)).x  #For me: .x to turn it into np.array
    
    return w
    
    
    
def compute_logistic_loss(X, y, theta):
    loss=0
    for i in range(y.size):
        #break
        s=y[i]*np.dot(X[i,:],theta)
        loss=loss-s/2+np.logaddexp(-s/2,s/2)  #I am adding log(1+exp(-s)) (to avoid overflow)
    #print(X[1,:].shape,theta.shape)
    loss=loss/y.size
    return loss
    
def validation_loss(X,y,w):
    loss_valid=compute_logistic_loss(X, y, w)
    #print('The loss on the validation data is:',loss_valid)
    return loss_valid
    
def plot_losses(X_train, Y_train, X_test, Y_test, plt_points=20):
    reg=np.zeros(plt_points)
    loss_train=np.zeros(plt_points)
    loss_test=np.zeros(plt_points)
    for i in range(plt_points):
        reg[i]=0.05*(i**2)   #x-axis is for regularization parameter
        w=fit_logistic_reg(X_train, Y_train, f_objective, l2_param=reg[i])
        loss_train[i]=validation_loss(X_train, Y_train, w)
        loss_test[i]=validation_loss(X_test, Y_test, w)
    plt.plot(reg, loss_train)
    plt.plot(reg, loss_test)
    plt.show()


def main():
    X_train=pd.read_csv('X_train.txt', delimiter=',').values
    X_test=pd.read_csv('X_val.txt', delimiter=',').values
    Y_train=pd.read_csv('y_train.txt', delimiter=',').values
    Y_test=pd.read_csv('y_val.txt', delimiter=',').values
     
    #Scaling all to [0, 1]
    X_train, X_test=feature_normalization(X_train, X_test)
    
    #Adding column of ones for bias
    X_train=np.c_[X_train, np.ones(X_train.shape[0])]
    X_test=np.c_[X_test, np.ones(X_test.shape[0])]
	
    #res=f_objective(theta, X, y, l2_param=1)
    #print(res)
    
    #w=fit_logistic_reg(X_train, Y_train, f_objective, l2_param=0.002)  #for large λ, w->0 but loss does not increase (???)
    #print('The loss on the training data is:', compute_logistic_loss(X_train, Y_train, w))
    #validation_loss(X_test, Y_test, w)
    
    plot_losses(X_train, Y_train, X_test, Y_test, plt_points=10) #test and validation losses are almost identical!!!
    #print(w)

if __name__ == "__main__":
    main()
