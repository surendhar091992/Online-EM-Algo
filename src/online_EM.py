"""
Implementation of Online EM Algorithm
Leon Zheng
"""

import numpy as np



# online_em2 (Titterington)
from src.mixture_poisson import poisson_theta_bar2
def online_EM2(data, init, gamma, save_iter_theta = False):
    all_theta =init.reshape((1,-1))
    theta = init
    iter_n = len(data)
    for i in range(iter_n):
        theta = poisson_theta_bar2(data[i],theta,gamma[i])
        if save_iter_theta:
            all_theta =np.concatenate([all_theta,theta.reshape((1,-1))],axis=0)
    if save_iter_theta:
        return theta, all_theta
    else:
        return theta

def online_EM_iter(new_data, s, theta, new_gamma, s_bar, theta_bar):
    # print(s_bar(new_data, theta).shape)
    # print(s.shape)
    new_s = s + new_gamma * (s_bar(new_data, theta) - s)
    #print(f'new_s = {new_s}')
    new_theta = theta_bar(new_s)
    #print(f'new_theta = {new_theta}')
    return new_s, new_theta


def online_EM(data, init, gamma, s_bar, theta_bar, save_iter_theta=False):
    all_theta = init.reshape((1,-1))
    s = 0
    theta = init
    for i in range(len(data)):
        if save_iter_theta:
            all_theta = np.concatenate([all_theta,theta.reshape((1,-1))],axis=0)
        # print(f'gamma = {gamma[i]}, Y_new = {data[i]}')
        s, theta = online_EM_iter(data[i], s, theta, gamma[i], s_bar, theta_bar)
    if save_iter_theta:
        return s, theta, all_theta
    else:
        return s, theta


def multi_online_EM(data, init, gamma_list, s_bar, theta_bar):
    N = len(gamma_list)
    all_theta_list = [init.reshape((1,-1))] * N
    s_list = [0] * N
    theta_list = [init] * N
    for i in range(len(data)):
        for j in range(N):
            all_theta_list[j] = np.concatenate([all_theta_list[j], theta_list[j].reshape((1,-1))], axis=0)
            s_list[j], theta_list[j] = online_EM_iter(data[i], s_list[j], theta_list[j], gamma_list[j][i], s_bar, theta_bar)
    return s_list, theta_list, all_theta_list


def polyak_ruppert_online_em(data, init, gamma, n_avg, s_bar, theta_bar):
    all_theta = init.reshape((1, -1))
    s = 0
    theta = init
    theta_hat_list = []
    n = len(data)
    for i in range(n):
        all_theta = np.concatenate([all_theta, theta.reshape((1, -1))], axis=0)
        s, theta_hat = online_EM_iter(data[i], s, theta, gamma[i], s_bar, theta_bar)
        if i < n_avg:
            theta = theta_hat
        else:
            theta_hat_list.append(theta_hat)
            theta = np.sum(np.array(theta_hat_list), axis=0) / len(theta_hat_list)
    return s, theta, all_theta
