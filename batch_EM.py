import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

# batch em for poisson mixture 

"""
Y: vector of (1,n), the whole data set 
theta: matrix of (2,m), theta of k-1 
output: P(W=j|y,theta) matrix (m,n) 
"""
def E_step(Y,theta):
    n = len(Y)
    m = theta.shape[1]
    w_y_theta = np.zeros((m,n))
    for i in range(m):
        w_y_theta[i,:] = theta[0,i]*theta[1,i]**Y*np.exp(-theta[1,i])
    sum_w_y_theta = np.sum(w_y_theta,axis=0)
    #print(sum_w_y_theta.shape)
    w_y_theta /= sum_w_y_theta
    #print(w_y_theta.shape)
    return w_y_theta

def M_step(Y,w_y_theta):
    n = len(Y)
    m = w_y_theta.shape[0]
    
    new_theta =np.zeros((2,m))
    for i in range(m):
        new_theta[0,i] = np.sum(w_y_theta[i,:])/n
        new_theta[1,i] = np.dot(Y,w_y_theta[i,:])/np.sum(w_y_theta[i,:])
    return new_theta

def batch_EM_iter(Y,theta):
    w_y_theta = E_step(Y,theta)
    new_theta = M_step(Y,w_y_theta)
    return new_theta

def batch_EM(Y,theta_init,max_iter=500):
    for i in range(max_iter):
        theta = batch_EM_iter(Y,theta_init)
        theta_init =theta
    return theta 