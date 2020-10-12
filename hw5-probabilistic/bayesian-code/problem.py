import matplotlib.pyplot as plt
import numpy.matlib as matlib
from scipy.stats import multivariate_normal
import numpy as np
import support_code
import math

def likelihood_func(w, X, y_train, likelihood_var):
    '''
    Implement likelihood_func. This function returns the data likelihood
    given f(y_train | X; w) ~ Normal(Xw, likelihood_var).

    Args:
        w: Weights
        X: Training design matrix with first col all ones (np.matrix)
        y_train: Training response vector (np.matrix)
        likelihood_var: likelihood variance

    Returns:
        likelihood: Data likelihood (float)
    '''

    #TO DO
    
    #First way (long way)
    A=y_train-np.transpose(np.dot(X, w))
    exponent=np.dot(np.transpose(A), A)/(2*likelihood_var)
    likelihood=(1/math.sqrt(2*math.pi* likelihood_var)**y_train.size)*np.exp(-exponent)
    return likelihood
    
    #Second way (better)
    #y_train = np.squeeze(np.asarray(y_train)) #to make it usable in multivariate_normal function
    #Gaussian_distr = multivariate_normal(mean=y_train, cov=likelihood_var)
    #yhat = X.dot(w)
    #return Gaussian_distr.pdf(yhat)



def get_posterior_params(X, y_train, prior, likelihood_var = 0.2**2):
    '''
    Implement get_posterior_params. This function returns the posterior
    mean vector \mu_p and posterior covariance matrix \Sigma_p for
    Bayesian regression (normal likelihood and prior).

    Note support_code.make_plots takes this completed function as an argument.

    Args:
        X: Training design matrix with first col all ones (np.matrix)
        y_train: Training response vector (np.matrix)
        prior: Prior parameters; dict with 'mean' (prior mean np.matrix)
               and 'var' (prior covariance np.matrix)
        likelihood_var: likelihood variance- default (0.2**2) per the lecture slides

    Returns:
        post_mean: Posterior mean (np.matrix)
        post_var: Posterior mean (np.matrix)
    '''

    # TO DO
    prior_mean=prior["mean"]
    prior_var=prior["var"]
    post_mean=np.dot( np.linalg.inv(np.dot(np.transpose(X), X)+likelihood_var*np.linalg.inv(prior_var) ) , np.dot(np.transpose(X), y_train)   )
    post_var=np.linalg.inv( np.dot(np.transpose(X), X)/likelihood_var+np.linalg.inv(prior_var) )
    
    return post_mean, post_var
    
    
    
def get_predictive_params(X_new, post_mean, post_var, likelihood_var = 0.2**2):
    '''
    Implement get_predictive_params. This function returns the predictive
    distribution parameters (mean and variance) given the posterior mean
    and covariance matrix (returned from get_posterior_params) and the
    likelihood variance (default value from lecture).

    Args:
        X_new: New observation (np.matrix object)
        post_mean, post_var: Returned from get_posterior_params
        likelihood_var: likelihood variance (0.2**2) per the lecture slides

    Returns:
        - pred_mean: Mean of predictive distribution
        - pred_var: Variance of predictive distribution
    '''

    # TO DO
    pred_mean=np.dot(np.transpose(post_mean), X_new)
    pred_var=np.dot(np.transpose(X_new), np.dot(post_var, X_new) ) +likelihood_var
    
    return pred_mean, pred_var

if __name__ == '__main__': #I did not do this

    '''
    If your implementations are correct, running
        python problem.py
    inside the Bayesian Regression directory will, for each sigma in sigmas_to-test generates plots
    '''

    np.random.seed(46134)
    actual_weights = np.matrix([[0.3], [0.5]])
    data_size = 40
    noise = {"mean":0, "var":0.2 ** 2}
    likelihood_var = noise["var"]
    xtrain, ytrain = support_code.generate_data(data_size, noise, actual_weights)

    #Question (b)
    sigmas_to_test = [1/2, 1/(2**5), 1/(2**10)]
    for sigma_squared in sigmas_to_test:
        prior = {"mean":np.matrix([[0], [0]]),
                 "var":matlib.eye(2) * sigma_squared}

        support_code.make_plots(actual_weights,   #WE HAVE A PROBLEM HERE
                                xtrain,
                                ytrain,
                                likelihood_var,
                                prior,
                                likelihood_func,
                                get_posterior_params,
                                get_predictive_params)
