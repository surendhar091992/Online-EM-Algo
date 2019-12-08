"""
Experiments - Online EM Algorithm with Mixture of Poisson
"""

from mixture_poisson import poisson_random_param, sample_poisson, poisson_s_bar, poisson_theta_bar
from online_EM import online_EM
import numpy as np

"""
Generating a problem instance
"""
# Parameters
n = 10  # Size of the data set
m = 3 # Number of clusters
max_l = 10 # Maximum possible value for parameter lambda of Poisson

# Random parameters
l, p = poisson_random_param(m, max_l) # Ground truth
theta_true = np.array([p, l])

# Data set
Y, W = sample_poisson(n, l, p)

"""
Online EM algo
"""
# Parameters
gamma_0 = 1
alpha = 1.
gamma = np.array([gamma_0 * np.power(n, -alpha) for l in range(1, n+1)])

# Initialization
p_init, l_init = poisson_random_param(m, max_l)
theta_init = np.array([p_init, l_init])

# Online EM algo
s, theta = online_EM(Y, theta_init, gamma, poisson_s_bar, poisson_theta_bar)

# Output
print("\n===============\nFinal results\n===============")
print(f"Truth:\n{theta_true}")
print(f"Online EM after {n} iterations:\n{theta}")