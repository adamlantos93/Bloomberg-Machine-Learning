import pandas as pd
import logging
import numpy as np
import sys
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import statistics
import random
from random import sample
from numpy.linalg import inv



#######################################
#### Normalization

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
	
	
    
########################################
#### The square loss function

def compute_square_loss(X, y, theta):
    """
    Given a set of X, y, theta, compute the square loss for predicting y with X*theta

    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        theta - the parameter vector, 1D array of size (num_features)

    Returns:
        loss - the square loss, scalar
    """
    loss = 0 #initialize the square_loss
    m=y.size
    dot1=np.dot(X,theta)-y
    loss=np.dot(dot1,dot1)/m
    return loss



########################################
### compute the gradient of square loss function
def compute_square_loss_gradient(X, y, theta):
    """
    Compute gradient of the square loss (as defined in compute_square_loss), at the point theta.
    Returns:
        grad - gradient vector, 1D numpy array of size (num_features)
    """
    grad=2*np.dot(np.dot(X,theta)-y,X)/y.size
    return grad



####################################
#### Batch Gradient Descent
def batch_grad_descent(X, y, alpha, num_iter=1000, check_gradient=False):
    """
    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        alpha - step size in gradient descent
        num_iter - number of iterations to run
        check_gradient - a boolean value indicating whether checking the gradient when updating

    Returns:
        theta_hist - store the history of parameter vector in iteration, 2D numpy array of size (num_iter+1, num_features)
                    for instance, theta in iteration 0 should be theta_hist[0], theta in ieration (num_iter) is theta_hist[-1]
        loss_hist - the history of objective function vector, 1D numpy array of size (num_iter+1)
    """
    num_instances, num_features = X.shape[0], X.shape[1]
    theta_hist = np.zeros((num_iter+1, num_features))  #Initialize theta_hist
    loss_hist = np.zeros(num_iter+1) #initialize loss_hist
    theta = np.zeros(num_features)+0.5 #initialize theta
    #TODO
    loss_hist[0]=compute_square_loss(X,y,theta)
    theta_hist[0,:]=theta
    error_threshold=1/10000
    n=0
    for i in range(num_iter):
        loss_hist[i+1]=compute_square_loss(X, y, theta_hist[i,:])
        grad=compute_square_loss_gradient(X, y, theta_hist[i,:])
        theta_hist[i+1,:]=theta_hist[i,:]-alpha*grad
        error=abs((loss_hist[i+1]-loss_hist[i]))/abs(loss_hist[i])
        if error < error_threshold and i > 2:
            print('The iteration is',i,'and the final loss percentage is',loss_hist[i+1])
            n=1
            final_iter=i
            break
    if n==0:
        final_iter=num_iter
        print('Theshold', error_threshold*100,'% did not pass; final loss percentage is',loss_hist[num_iter])
    return theta_hist, loss_hist, final_iter


###################################################
### Compute the gradient of Regularized Batch Gradient Descent
def compute_regularized_square_loss_gradient(X, y, theta, lambda_reg):
    """
    Compute the gradient of L2-regularized square loss function given X, y and theta

    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        theta - the parameter vector, 1D numpy array of size (num_features)
        lambda_reg - the regularization coefficient

    Returns:
        grad - gradient vector, 1D numpy array of size (num_features)
    """
    grad=2*np.dot(np.dot(X,theta)-y,X)/y.size+2*lambda_reg*theta
    return grad

###################################################
### Batch/Full Gradient Descent with regularization term
def regularized_grad_descent(X, y, alpha=0.1, lambda_reg=1, num_iter=1000, check_gradient=False):
    """
    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        alpha - step size in gradient descent
        lambda_reg - the regularization coefficient
        numIter - number of iterations to run

    Returns:
        theta_hist - the history of parameter vector, 2D numpy array of size (num_iter+1, num_features)
        loss_hist - the history of loss function without the regularization term, 1D numpy array.
    """
    (num_instances, num_features) = X.shape
    theta = np.zeros(num_features) #Initialize theta
    theta_hist = np.zeros((num_iter+1, num_features))  #Initialize theta_hist
    loss_hist = np.zeros(num_iter+1) #Initialize loss_hist
    #TODO
    loss_hist[0]=compute_square_loss(X,y,theta)
    theta_hist[0,:]=theta
    error_threshold=1/100000
    n=0
    for i in range(num_iter):
        grad=compute_regularized_square_loss_gradient(X, y, theta_hist[i,:],lambda_reg)
        theta_hist[i+1,:]=theta_hist[i,:]-alpha*grad
        loss_hist[i+1]=compute_square_loss(X, y, theta_hist[i,:])
        error=abs((loss_hist[i+1]-loss_hist[i]))/abs(loss_hist[i])
        if error < error_threshold and i>1:
            print('The iteration is',i,'and the final loss percentage is',loss_hist[i+1])
            n=1
            final_iter=i
            break
    if n==0:
        final_iter=num_iter
        print('Theshold', error_threshold*100,'% did not pass; final loss percentage is',loss_hist[num_iter])
    return theta_hist, loss_hist, final_iter
    


#############################################
### Stochastic/Minibatch Gradient Descent
def minibatch_grad_descent(X, y, alpha=0.1, lambda_reg=1, epochs=1000, batch_size=20):
    """
    In this question you will implement stochastic gradient descent with a regularization term
                NOTE: In SGD, it's not always a good idea to use a fixed step size. Usually it's set to 1/sqrt(t) or 1/t
                if alpha is a float, then the step size in every iteration is alpha.
                if alpha == "1/sqrt(t)", alpha = 1/sqrt(t)
                if alpha == "1/t", alpha = 1/t
        lambda_reg - the regularization coefficient
        epochs - number of epochs (i.e number of times) to go through the whole training set
        num_iter - times to perform gradient descent with the corresponding batch of the epoch
    """
    num_instances, num_features = X.shape[0], X.shape[1]
    theta = np.ones(num_features) #Initialize theta

    theta_hist = np.zeros((epochs+1, num_features))  #Initialize theta_hist
    loss_hist = np.zeros((epochs+1)) #Initialize loss_hist
    #TODO
    X_batch=np.zeros((batch_size, num_features))  #for batch_size=1, we perform stochastic GD
    y_batch=np.zeros((batch_size, num_features))
    loss_hist[0]=compute_square_loss(X,y,theta)
    theta_hist[0,:]=theta
    error_threshold=1/1000
    n=0
    for j in range(epochs):  #at every epoch, we select a different batch
        sample=random.sample(range(0,num_instances),batch_size) #at every epoch, we cycle through batches chosen randomly
        X_batch=X[sample,:]
        y_batch=y[sample]
        for i in range(60): #60 steps per batch
            grad=compute_regularized_square_loss_gradient(X_batch, y_batch, theta,lambda_reg)
            alpha=1/(i+1) #variable step size
            theta=theta-alpha*grad 
        theta_hist[j+1,:]=theta 
        loss_hist[j+1]=compute_square_loss(X_batch, y_batch, theta)
        error=(loss_hist[j+1]-loss_hist[j])/abs(loss_hist[j])
        if abs(error) < error_threshold and error < 0 and j > 2:  #Convergence criterion: small difference between loss[j+1] and loss[j] and want final result to be lowest than the one previous
            print('The epoch is',j,'and the final loss percentage is',loss_hist[j+1])
            final_epoch=j
            n=1
            break
    if n==0:
        print('Theshold', error_threshold*100,'% did not pass; final loss percentage is',loss_hist[epochs])
        final_epoch=epochs
    return theta_hist, loss_hist, final_epoch
   
   
def binarize_data(y): #takes data and maps it to {-1,1}
	for i in range(y.size):
		if y[i]>=0:
			y[i]=1
		else:
			y[i]=-1
	return y

def compute_perceptron_loss(X, y, theta):
    loss=0
    for i in range(y.size):
        A=max(0,-y[i]*np.dot(theta,X[i,:]))
        loss=loss+A
    loss=loss/y.size
    return loss
	
   
#############################################
### Perceptron algorithm
def regularized_perceptron(X, y, lambda_reg=0.1, epochs=100):
    num_instances, num_features = X.shape[0], X.shape[1]
    theta = np.zeros(num_features) #Initialize theta

    theta_hist = np.zeros((epochs+1, num_features))  #Initialize theta_hist
    loss_hist = np.zeros((epochs+1)) #Initialize loss_hist
    
    y=binarize_data(y) #turn it into a classification problem
    loss_hist[0]=compute_square_loss(X,y,theta)
    theta_hist[0,:]=theta
    for j in range(epochs):  #End when hyperplane w*x=0 separates the data
        m=1 #we find the hyperplane when at the end of an epoch we get m=1
        for i in range(num_instances): #cycling through data
            if y[i]*np.dot(X[i,:],theta) <= 0:
                subgrad=y[i]*X[i,:]
                theta=theta+subgrad-2*lambda_reg*theta
                m=0
        theta_hist[j+1,:]=theta 
        loss_hist[j+1]=compute_perceptron_loss(X, y, theta)
        if m==1:
            final_epoch=j
            print('We found the hyperplane that separates the two classes.')
            print('Final loss is',loss_hist[j+1])
            break
    if m==0:
        print('After',epochs,'epochs, we did not find a hyperplane that separates the two classes.\n')
        print('Final loss is:', loss_hist[epochs])
        final_epoch=epochs
    return theta_hist, loss_hist, final_epoch




def coordinate_descent_lasso(X, y, lambda_reg=0.1, num_iter=1000):   #cyclic coordinate descent
    (num_instances, num_features) = X.shape    #instances are samples
    
    #To initialize the weights, we use the "solution" of the ridge regression:
    A=np.dot(np.transpose(X),X) #gives a dxd matrix
    inverse=inv(A+lambda_reg*np.identity(num_features, dtype=None))
    theta=np.dot(inverse,np.dot(np.transpose(X),y))
    
    theta_hist = np.zeros((num_iter+1, num_features))  #Initialize theta_hist
    loss_hist = np.zeros(num_iter+1) #Initialize loss_hist
    loss_hist[0]=compute_square_loss(X,y,theta)
    theta_hist[0,:]=theta
    
    error_threshold=1/100000
    n=0
    
    for i in range(num_iter):
        for j in range(num_features):
            B=np.dot(np.transpose(X),np.dot(X,theta)) #vector in R^d
            C=np.dot(np.transpose(X),y)  #vector in R^d
            #definition of main variables of closed-form solution:
            aj=2*A[j,j]
            cj=2*(C[j]-B[j]+A[j,j]*theta[j])
            
            p=abs(cj)-lambda_reg
            if p >0:
                theta[j]=(cj/aj-lambda_reg*cj*abs(aj/cj)/aj**2)
            else:
                theta[j]=0
            
        loss_hist[i+1]=compute_square_loss(X, y, theta)
        theta_hist[i+1,:]=theta
        
        #Convergence criterion (every time we go through all components of the weight vector
        error=abs((loss_hist[i+1]-loss_hist[i]))/abs(loss_hist[i])
        if i > 1 and error < error_threshold:
            print('The iteration is',i,'and the final loss percentage is',loss_hist[i+1])
            n=1 #to check if it converged or we just reached the end of the iterations
            final_iter=i
            break
            
    if n==0:
        print('Theshold', error_threshold*100,'% did not pass; final loss percentage is',loss_hist[num_iter])
        final_iter=num_iter
    #print(theta_hist[final_iter,:],'lala') #to get results for weights
    return theta_hist, loss_hist, final_iter
    

def validation_square_loss(X, y, w):
	loss_valid=compute_square_loss(X, y, w)
	print('The square loss on the validation data is:',loss_valid)

def validation_perceptron_loss(X,y,w):
    y=binarize_data(y)
    loss_valid=compute_perceptron_loss(X, y, w)
    print('The square loss on the validation data is:',loss_valid)

def main():
    #Loading the dataset.
    df = pd.read_csv('data.csv', delimiter=',')
    X = df.values[:,:-1]
    y = df.values[:,-1]

    #Split into Train and Test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =100, random_state=10)

    #Scaling all to [0, 1]
    X_train, X_test = feature_normalization(X_train, X_test)
    
    B=5 # Add bias term with B being the multiplier of b, to lessen the effect of regularization on the bias term
    X_train = np.hstack((X_train, np.ones((X_train.shape[0], 1))*B))  #Add bias term
    X_test = np.hstack((X_test, np.ones((X_test.shape[0], 1))*B)) # Add bias term 
    
    #results=regularized_grad_descent(X_train, y_train, alpha=0.01, lambda_reg=0.05, num_iter=1000, check_gradient=False)
    #validation_square_loss(X_test, y_test, w)
    
    #results=minibatch_grad_descent(X_train, y_train, alpha=0.01, lambda_reg=0.05, epochs=1000, batch_size=32)
    #print('The losses for the final 4 epochs are:', results[1][results[2]-2:results[2]+1])
    #validation_square_loss(X_test, y_test, w)
    
    #results=coordinate_descent_lasso(X_train, y_train, lambda_reg=0.5, num_iter=1000)
    #print('The losses for the final 4 iterations on the training set are:', results[1][results[2]-2:results[2]+2])
    #validation_square_loss(X_test, y_test, w)
    
    results=regularized_perceptron(X_train, y_train, lambda_reg=0.03, epochs=5000)
    w=results[0][results[2]][:]
    #print(w)
    validation_perceptron_loss(X_test, y_test, w)
    

if __name__ == "__main__":
    main()
