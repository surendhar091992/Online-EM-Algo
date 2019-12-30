"""
Convergence rate experiements
Leon Zheng
"""

from mixture_poisson import sample_poisson, poisson_random_param, poisson_s_bar, poisson_theta_bar
from online_EM import multi_online_EM
import matplotlib.pyplot as plt
import numpy as np

def change_order(theta):
    order = np.argsort(theta[1,:])
    new_theta = np.zeros(theta.shape)
    for i in range(m):
        new_theta[0,i] = theta[0,order[i]]
        new_theta[1,i] = theta[1,order[i]]
    return new_theta

# Create data set
n = 5000  # Size of the data set
m = 3
max_l = 100
l = [10, 40, 80]
p = [0.4, 0.35, 0.25]
theta_true = np.array([p, l])
print('Creating data set...')
Y, W = sample_poisson(n, l, p)
print('Data set created!\n')

print('Online EM algorithm for Poisson mixture...')
# Initialization of parameters for online EM
p_init = np.ones(m)/m + 0.01 * np.random.rand(3)
l_init = max_l * np.ones(m)/m + np.random.rand(3)
theta_init = np.array([p_init, l_init])
print(f'Initial parameters: \n{theta_init}')

# Step size for EM algo
gamma_0 = 1
alphas = [1., 0.6]
N = len(alphas)
gamma_list = [np.array([gamma_0 * np.power(l, -alphas[j]) for l in range(1, n + 1)]) for j in range(N)]

# Online EM algo
s_list, theta_list, all_theta_list = multi_online_EM(Y, theta_init, gamma_list, poisson_s_bar, poisson_theta_bar)
print('Done!')
print('\n============Results============')
print(f"Truth:\n{theta_true}")
for j in range(N):
    print(f"Online EM with alpha {alphas[j]} after {n} iterations:\n{theta_list[j]}")
print(all_theta_list[0].shape)

# Visualize the process of parameter convergence
names = [['$\omega_1$', '$\lambda_1$'], ['$\omega_2$', '$\lambda_2$'], ['$\omega_3$', '$\lambda_3$']]
# truth = change_order(theta_true).reshape((1, -1))[0]
order_list = [np.argsort(theta_list[j][1, :]) for j in range(N)]
print(f'order: {order_list}')
#
# order = np.concatenate([order, order + 3])
fig, axes = plt.subplots(3, 2, sharex=True, figsize=(10, 5))
for i in range(3):
    for om_la in range(2):
        for j in range(N):
            axes[i, om_la].plot(all_theta_list[j][:, order_list[j][i] + 3 * om_la], label=f'alpha={alphas[j]}', linewidth = 1 if j == 0 else 0.8)
        axes[i, om_la].plot([0, n], [theta_true[om_la, i], theta_true[om_la, i]], 'r--', label='Truth')
        if i == 2:
            axes[i, om_la].set_xlabel('Samples')
        axes[i, om_la].set_ylabel(names[i][om_la])
        if om_la == 1 and i == 0:
            axes[i, om_la].legend()
        axes[i, om_la].grid()
plt.savefig('cv_rate.png')
