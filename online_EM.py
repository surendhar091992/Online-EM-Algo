"""
Implementation of Online EM Algorithm
Leon Zheng
"""

def online_EM_iter(new_data, s, theta, new_gamma, s_bar, theta_bar):
    new_s = s + new_gamma * (s_bar(new_data, theta) - s)
    print(f'new_s = {new_s}')
    new_theta = theta_bar(new_s)
    print(f'new_theta = {new_theta}')
    return new_s, new_theta


def online_EM(data, init, gamma, s_bar, theta_bar):
    s = 0
    theta = init
    iter = len(data)
    for i in range(iter):
        s, theta = online_EM_iter(data[i], s, theta, gamma[i], s_bar, theta_bar)
    return s, theta
