"""
Comparing Online EM and Batch EM algorithm
Leon Zheng
"""


from mixture_poisson import sample_poisson, poisson_s_bar, poisson_theta_bar, poisson_random_param, change_order
from online_EM import online_EM
from batch_EM import batch_EM
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# PARAMETERS
num_exp = 500 #
max_iteration_1 = 3 # For batch EM
max_iteration_2 = 5 # For batch EM
n = 10000  # Size of the data set
m = 3
max_l = 100
l = [10, 40, 80]
p = [0.4, 0.35, 0.25]

# Create data set
theta_true = np.array([p, l])
print('Creating data set...')
Y, W = sample_poisson(n, l, p)
print('Data set created!\n')

# Random initilization
init_list = []
for i in range(num_exp):
    p_init, l_init = poisson_random_param(m, max_l)
    theta_init = np.array([p_init, l_init])
    init_list.append(theta_init)

# Online EM (alpha = 1)
gamma_0 = 1
alpha = 1.
gamma = np.array([gamma_0 * np.power(l, -alpha) for l in range(1, n+1)])
all_theta_1 =[]
print(f'Online EM algorithm, alpha = {alpha}...')
for i in range(num_exp):
    if i%(num_exp//10) == 0:
        print(f'Experience {i+1}/{num_exp}')
    theta_init = init_list[i]
    s, theta = online_EM(Y, theta_init, gamma, poisson_s_bar, poisson_theta_bar)
    theta = change_order(theta)
    all_theta_1.append(theta.reshape(1,-1)[0])
print('Done!\n')

# Online EM (alpha =0.6)
gamma_0 = 1
alpha = 0.6
gamma = np.array([gamma_0 * np.power(l, -alpha) for l in range(1, n+1)])
all_theta_06 =[]
print(f'Online EM algorithm, alpha = {alpha}...')
for i in range(num_exp):
    if i%(num_exp//10) == 0:
        print(f'Experience {i+1}/{num_exp}')
    theta_init = init_list[i]
    s, theta = online_EM(Y, theta_init, gamma, poisson_s_bar, poisson_theta_bar)
    theta = change_order(theta)
    all_theta_06.append(theta.reshape(1,-1)[0])
print('Done!\n')

# Batch EM 1
gamma_0 = 1
alpha = 1.
gamma = np.array([gamma_0 * np.power(l, -alpha) for l in range(1, n+1)])
all_theta_batch_1 =[]
print(f'Batch EM algorithm, max_iter = {max_iteration_1}...')
for i in range(num_exp):
    if i%(num_exp//10) == 0:
        print(f'Experience {i+1}/{num_exp}')
    theta_init = init_list[i]
    theta = batch_EM(Y, theta_init, max_iter = max_iteration_1)
    theta = change_order(theta)
    all_theta_batch_1.append(theta.reshape(1,-1)[0])
print('Done!\n')

# Batch EM 2
gamma_0 = 1
alpha = 1.
gamma = np.array([gamma_0 * np.power(l, -alpha) for l in range(1, n+1)])
all_theta_batch_2 =[]
print(f'Batch EM algorithm, max_iter = {max_iteration_2}...')
for i in range(num_exp):
    if i%(num_exp//10) == 0:
        print(f'Experience {i+1}/{num_exp}')
    theta_init = init_list[i]
    theta = batch_EM(Y, theta_init, max_iter = max_iteration_2)
    theta = change_order(theta)
    all_theta_batch_2.append(theta.reshape(1,-1)[0])
print('Done!\n')

# Stacking all thetas
all_theta_1 = np.stack(all_theta_1,axis=0)
all_theta_06 = np.stack(all_theta_06,axis=0)
all_theta_batch_1 = np.stack(all_theta_batch_1,axis=0)
all_theta_batch_2 = np.stack(all_theta_batch_2,axis=0)
true_params = change_order(theta_true).reshape((1,-1))[0]
all_theta = np.concatenate([all_theta_1, all_theta_06, all_theta_batch_1, all_theta_batch_2],axis=0)

# Plotting
methods = ['OL1'] * num_exp + ['OL06'] * num_exp + [f'EM{max_iteration_1}'] * num_exp + [f'EM{max_iteration_2}'] * num_exp
params = ['$\omega_1$','$\omega_2$','$\omega_3$', '$\lambda_1$', '$\lambda_2$', '$\lambda_3$']
df = pd.DataFrame(all_theta, columns=params)
df['Methods'] = methods
count = 0
fig, axes = plt.subplots(2, 3, figsize=(10,5))
fig.subplots_adjust(wspace = 0.3, hspace=0.3)
for i in range(2):
    for j in range(3):
        param = params[3 * i + j]
        sns.boxplot(x='Methods', y=param, data=df, ax = axes[i, j])
        axes[i, j].plot([-0.5,3.5],[true_params[count],true_params[count]],'--')
        count += 1
plt.savefig('methods.png')