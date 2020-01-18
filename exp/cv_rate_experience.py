"""
Convergence rate experiements
Leon Zheng
"""

from src.mixture_poisson import sample_poisson, poisson_s_bar, poisson_theta_bar
from src.online_EM import multi_online_EM, polyak_ruppert_online_em
import matplotlib.pyplot as plt
import numpy as np



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
gamma_0 = [1, 1]
alphas = [1., 0.6]
N = len(alphas)
gamma_list = [np.array([gamma_0[j] * np.power(l, -alphas[j]) for l in range(1, n + 1)]) for j in range(N)]

# Online EM algo
s_list, theta_list, all_theta_list = multi_online_EM(Y, theta_init, gamma_list, poisson_s_bar, poisson_theta_bar)
print('Done!')
print('\n============Results============')
print(f"Truth:\n{theta_true}")
for j in range(N):
    print(f"Online EM with alpha {alphas[j]} after {n} iterations:\n{theta_list[j]}")
print(all_theta_list[0].shape)

# Polyak-Ruppert averaging
gamma_0 = 1
alpha = 0.6
gamma = np.array([gamma_0 * np.power(l, -alpha) for l in range(1, n + 1)])
n_avg = n // 2
s_polyak, theta_polyak, all_theta_polyak = polyak_ruppert_online_em(Y, theta_init, gamma, n_avg, poisson_s_bar, poisson_theta_bar)
order_polyak = np.argsort(theta_polyak[1, :])

# Visualize the process of parameter convergence
names = [['$\omega_1$', '$\lambda_1$'], ['$\omega_2$', '$\lambda_2$'], ['$\omega_3$', '$\lambda_3$']]
# truth = change_order(theta_true).reshape((1, -1))[0]
order_list = [np.argsort(theta_list[j][1, :]) for j in range(N)]
print(f'order: {order_list}')

# Plotting
fig, axes = plt.subplots(3, 2, sharex=True, figsize=(10, 5))
for i in range(3):
    for om_la in range(2):
        for j in range(N):
            axes[i, om_la].plot(all_theta_list[j][:, order_list[j][i] + 3 * om_la], '-' if j == 0 else ':', label=f'alpha={alphas[j]}', color='orange' if j==1 else 'blue', linewidth = 1 if j == 0 else 1)
        axes[i, om_la].plot(all_theta_polyak[:, order_polyak[i] + 3 * om_la], label=f'alpha=0.6 + Avg', color='red', linewidth = 1)
        axes[i, om_la].plot([0, n], [theta_true[om_la, i], theta_true[om_la, i]], 'r--', label='Truth', color = 'green')
        if i == 2:
            axes[i, om_la].set_xlabel('Samples')
        axes[i, om_la].set_ylabel(names[i][om_la])
        if om_la == 1 and i == 0:
            axes[i, om_la].legend()
        axes[i, om_la].grid()
plt.savefig('cv_rate.png')
plt.show()