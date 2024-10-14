import sys, os
import numpy as np
cur_path = os.path.dirname(os.path.realpath(__file__))
module_path = os.path.abspath(os.path.join(cur_path, '..'))

# load environment
sys.path.insert(0, module_path)
from tm_avoidance_model import Env
from utils import simulate_tokamak_data


if __name__ == '__main__':
    tm_avoidance_dir = os.path.join(module_path, 'tm_avoidance_model')

    # x0 metrics
    x0_mean_path = os.path.join(tm_avoidance_dir, 'x0_mean.npy')
    x0_mean = np.load(x0_mean_path)
    x0_std_path = os.path.join(tm_avoidance_dir, 'x0_std.npy')
    x0_std = np.load(x0_std_path)
    print('x0 mean: {}\nx0 std: {}'.format(x0_mean.shape, x0_std.shape))

    # x1 metrics
    x1_mean_path = os.path.join(tm_avoidance_dir, 'x1_mean.npy')
    x1_mean = np.load(x1_mean_path)
    x1_std_path = os.path.join(tm_avoidance_dir, 'x1_std.npy')
    x1_std = np.load(x1_std_path)

    N = 10000
    simulated_x0, simulated_x1 = simulate_tokamak_data(N, x0_mean, x0_std, x1_mean, x1_std)
    env = Env(x0_data=simulated_x0, x1_data=simulated_x1)
    s = env.reset()
    action = np.array([0.5, 0.5, 0.5])
    s_prime, reward, done, info = env.step(action)
    print("reward", reward)
    print("Episode Terminated:", done)
