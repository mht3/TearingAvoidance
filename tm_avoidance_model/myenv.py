import math
import gymnasium as gym
import numpy as np
from keras import models
import os

def f1_m(y_true, y_pred):
    return 1.0

def sigmoid(z):
    return 1/(1+math.e**(-z))

# Predictor setting
cur_path = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.abspath(os.path.join(cur_path, '..', 'data'))
predictor_dir = os.path.abspath(os.path.join(cur_path, '..', 'tm_prediction_model'))
n_predictor = 5

# Action setting
low_action, high_action = [-1, -1, -1], [1, 1, 1]
clip_action1 = [0, 0, 0]
clip_action2 = [10000, 1, 1]
threshold = 0.5

# Normalization
s_mean = np.array([3.977, 1.587, 0.5103, 23303., 42.93]) # ne, te, 1/q, pres, rot
s_std = np.array([2.764, 1.560, 0.4220, 35931., 58.47])
a_mean = np.array([4072.8761393229165, 0.6, 0.41])
a_std = np.array([3145.5935872395835, 0.31, 0.31])

class Env(gym.Env):
    def __init__(self, x0_data=None, x1_data=None, save_dataset_metrics=False):
        '''
        Updated gym environment for "Avvoiding fusion plasma tearing instability with deep reinforcement learning" by Seo et al.
        Updates: 
            1. Packaged the environment class (see __init__.py)
            2. Generalized the environment to work when being called from any directory using os.path.join
            3. Added optional arguments for the constructor so that simulated data can be passed in place of real DIII-D data

        X0 Labels (0D params):
            ['bt', 'ip', 'pinj', 'tinj', 'R0_EFITRT1', 'kappa_EFITRT1', 'tritop_EFIT01', 'tribot_EFIT01', 'gapin_EFIT01', 'ech_pwr_total', 'EC.RHO_ECH'] # at t+dt
        X1 Labels (1D profiles):
            ['thomson_density_mtanh_1d', 'thomson_temp_mtanh_1d', '1/qpsi_EFITRT1', 'pres_EFIT01', 'cer_rot_csaps_1d'] # at t
        dynamic model output (used for reward):
            ['betan_EFITRT1', 'tm_label'] # at t+dt
        policy output (action): 
            [beam power, top triangularity, bottom triangularity]
        '''
        super(Env, self).__init__()
        # Load data and models

        # Requires access to DIII-D data
        if x0_data is None:
            x0_data = os.path.join(data_dir, 'x0.npy')
            self.x0 = np.load(x0_data)
        else:
            # assume data is directly passed as numpy array
            self.x0 = x0_data

        if x1_data is None:
            x1_data = os.path.join(data_dir, 'x1.npy')
            self.x1 = np.load(x1_data)
        else:
            # assume data is directly passed as numpy array
            self.x1 = x1_data

        self.x0_mean, self.x0_std = self.x0.mean(axis=0).astype(np.float32), self.x0.std(axis=0).astype(np.float32)
        self.x1_mean, self.x1_std = self.x1.mean(axis=0).astype(np.float32), self.x1.std(axis=0).astype(np.float32)
        self.predictors = [models.load_model(os.path.join(predictor_dir, f'best_model_{i}'), custom_objects={'f1_m':f1_m}) for i in range(n_predictor)]

        if save_dataset_metrics:
            # Save normalizing factors
            np.save('x0_mean.npy', self.x0_mean)
            np.save('x0_std.npy', self.x0_std)
            np.save('x1_mean.npy', self.x1_mean)
            np.save('x1_std.npy', self.x1_std)

        # Balance dataset
        yy = np.mean([p.predict([self.x0, self.x1]) for p in self.predictors], axis=0)
        idx_pos = (sigmoid(yy[1]) > threshold)
        x0_pos, x1_pos = self.x0[idx_pos].copy(), self.x1[idx_pos].copy()

        # ratio of 0 tearability to 1 tearability predictions. ex: 5:1
        balance_ratio = (len(idx_pos) - sum(idx_pos)) // sum(idx_pos)
        for _ in range(balance_ratio - 1):
            self.x0 = np.append(self.x0, x0_pos, axis=0)
            self.x1 = np.append(self.x1, x1_pos, axis=0)

        # Setting for RL
        self.action_space = gym.spaces.Box(
            low = np.array(low_action),
            high = np.array(high_action),
            dtype = np.float32
        )
        self.observation_space = gym.spaces.Box(
            low = -2 * np.ones_like(self.x1_mean), #self.x1_mean - 2 * self.x1_std,
            high = 2 * np.ones_like(self.x1_mean), #self.x1_mean + 2 * self.x1_std,
            dtype = np.float32
        )
        
        # Initialize
        self.episodes = 0
        self.reset()

    def reset(self):
        self.episodes += 1
        self.i_model = np.random.randint(n_predictor)
        self.idx = np.random.randint(len(self.x0))
        return (self.x1[self.idx] - s_mean) / s_std

    def step(self, action):
        # Take action

        # action is beam power (W), top triangularity, and bottom triangularity
        action1 = np.clip(action * a_std + a_mean, clip_action1, clip_action2)
        # creates copies of the arrays with shapes 1x11 and 1x33x5 
        x0_tmp, x1_tmp = self.x0[[self.idx]].copy(), self.x1[[self.idx]].copy()
        # update 
        x0_tmp[0, 2] = action1[0]
        x0_tmp[0, 3] = min(1.0, action1[0] * self.x0_mean[3] / self.x0_mean[2])
        # update top triangularity
        x0_tmp[0, 6] = action1[1]
        # update bottom triangularity
        x0_tmp[0, 7] = action1[2]

        # Predict next step
        y = self.predictors[self.i_model].predict([x0_tmp, x1_tmp])
        betan, tearability = y[0][0], sigmoid(y[1][0])

        # Estimate reward
        if tearability < threshold:
            reward = betan
        else:
            reward = threshold - tearability
        # print(self.episodes, action[0], betan, tearability, reward)
        return (self.x1[self.idx] - s_mean) / s_std, reward, True, {}

    def render(self, mode = 'human'):
        pass

    def close(self):
        pass
