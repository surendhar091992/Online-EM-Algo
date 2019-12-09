"""
Implementation of data set generator - Mixture of Poisson
Leon Zheng
"""

import numpy as np


"""
Constructing a data set, mixture of Poisson
"""
def sample_from_discrete(dist):
    p = np.cumsum(dist)
    u = np.random.rand()
    i = 0
    while i < len(p) and u > p[i]:
        i += 1
    return i

def poisson_random_param(m, max_l):
    """
    Generates random parameters for Poisson mixture. m is the number of clusters. Generetas m random Poisson parameters in ]0; max_l].
    :param m: number of clusters
    :param max_l: maximum value for Poisson parameter
    :return: vector (m, ), vector (m, )
    """
    l = max_l * np.random.rand(m)
    p = np.random.rand(m)
    p /= np.sum(p)
    # print(f'lambda = {l}')
    # print(f'p = {p}')
    return np.array(p), np.array(l)

def sample_poisson(n, l, p):
    W = []
    Y = []
    for i in range(n):
        w = sample_from_discrete(p)
        W.append(w)
        Y.append(np.random.poisson(l[w]))
    # print('Sampling')
    # print(f'W = {W}')
    # print(f'Y = {Y}')
    return np.array(Y), np.array(W)


"""
Poisson Mixture - Fonctions for Online EM
Representation:
 * theta.shape = (2, m). First line are omega_j, second are lambda_j, for j in [1, m].
 * s.shape = (m, 2). For each line j in [1, m], we represente S_j(y, w) in page 7.
"""
def poisson_s_bar(Y, theta):
    m = theta.shape[1]
    s_bar = np.zeros((m, 2))
    # E-step
    # print(f'y in poisson_s_bar = {Y}')
    w_y_theta = np.array([theta[0, j] * theta[1, j] ** Y * np.exp(- theta[1, j]) for j in range(m)])
    w_y_theta /= np.sum(w_y_theta)
    print(f'w_y_theta = {w_y_theta}')
    for j in range(m):
        s_bar[j, 0] = w_y_theta[j]
        s_bar[j, 1] = Y * w_y_theta[j]
    return s_bar

def poisson_theta_bar(new_s):
    m = new_s.shape[0]
    theta_bar = np.zeros((2, m))
    for j in range(m):
        theta_bar[0, j] = new_s[j, 0]
        theta_bar[1, j] = new_s[j, 1] / new_s[j, 0]
    return theta_bar

"""
Poisson Mixture - Fonctions for Classical EM
"""
def poisson_theta_bar2(y,theta,gamma):
    m= theta.shape[1]
    theta_bar = np.zeros((2,m))
    # E-step
    w_y_theta = np.array([theta[0, j] * theta[1, j] ** y * np.exp(- theta[1, j]) for j in range(m)])
    w_y_theta /= np.sum(w_y_theta)
    # update for omega 
    for j in range(m):
        theta_bar[0,j] = theta[0,j] + gamma*(w_y_theta[j]-theta[0,j])
        theta_bar[1,j] = theta[1,j] + gamma*w_y_theta[j]*(y-theta[1,j])/theta[0,j]
    return theta_bar

"""
Main - Testing sampling for constructing the data set
"""
if __name__ == '__main__':
    n = 50  # Size of the data set
    m = 3
    max_l = 10
    l, p = poisson_random_param(m, max_l)
    # Q = np.zeros(m)
    # for i in range(100000):
    #     idx = sample_from_discrete(p)
    #     Q[idx] += 1
    # print(Q / 100000)

    Y, W = sample_poisson(n, l, p)
    print(f'lambda = {l}')
    print(f'p = {p}')
    print('Sampling')
    print(f'W = {W}')
    print(f'Y = {Y}')
