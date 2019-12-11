"""
Implementation of Online EM Algorithm
Leon Zheng
"""

import numpy as np

def online_EM_iter(new_data, s, theta, new_gamma, s_bar, theta_bar):
    new_s = s + new_gamma * (s_bar(new_data, theta) - s)
    #print(f'new_s = {new_s}')
    new_theta = theta_bar(new_s)
    #print(f'new_theta = {new_theta}')
    return new_s, new_theta


def online_EM(data, init, gamma, s_bar, theta_bar, save_iter_theta=False):
    all_theta =init.reshape((1,-1))
    s = 0
    iter = len(data)
    for i in range(iter):
        # print(f'gamma = {gamma[i]}, Y_new = {data[i]}')
        s, theta = online_EM_iter(data[i], s, theta, gamma[i], s_bar, theta_bar)
        if save_iter_theta:
            all_theta =np.concatenate([all_theta,theta.reshape((1,-1))],axis=0)
    if save_iter_theta:
        return s, theta, all_theta
    else:
        return s, theta

# online_em2 (Titterington)
from mixture_poisson import poisson_theta_bar2
def online_EM2(data, init, gamma, save_iter_theta = False):
    all_theta =init.reshape((1,-1))
    theta =init
    iter = len(data)
    for i in range(iter):
        theta = poisson_theta_bar2(data[i],theta,gamma[i])
        if save_iter_theta:
            all_theta =np.concatenate([all_theta,theta.reshape((1,-1))],axis=0)
    if save_iter_theta:
        return theta, all_theta
    else:
        return theta