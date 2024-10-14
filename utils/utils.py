import os, sys
import numpy as np

cur_path = os.path.dirname(os.path.realpath(__file__))
module_path = os.path.join(cur_path, '..')
sys.path.insert(0, module_path)


def simulate_tokamak_data(n, mu_0, sigma_0, mu_1, sigma_1):
    '''
    Simulate DIII-D data based upon calculated mean and standard deviations of x0 and x1 from Seo et al.
    Parameters:
        n (int):
            total number of samples
        mu_0 (np.array):
            mean of 0D parameters from dataset. Shape = (11,)
        sigma_0 (np.array):
            standard deviation of 0D parameters from dataset. Shape = (11,)
        mu_1 (np.array):
            mean of plasma profile from dataset. Shape = (33, 5)
        sigma_1 (np.array):
            standard deviation of 0D parameters from dataset. Shape = (33, 5)
    '''
    x0 = np.random.normal(loc=mu_0, scale=sigma_0, size=(n, mu_0.shape[0]))
    x1 = np.random.normal(loc=mu_1, scale=sigma_1, size=(n, mu_1.shape[0], mu_1.shape[1]))
    return x0, x1