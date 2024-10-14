
import torch
import numpy as np
from torch.utils.data import Dataset

class LyapunovDataset(Dataset):
    def __init__(self, filename, state_dim=5, action_dim=3):
        '''
        PyTorch dataloader for trajectory data from simulator and policy.
        __getitem__ returns state, action, next_state pairs.
        Parameters:
        filename (str):
            trajectories.npz file generated from lyapunov_dataset.py
        state_dim (int):
            number of states
        action_dim (int):
            number of actions
        '''
        super().__init__()
        self.data = self.load_torch(filename)
        self.state_dim = state_dim
        self.action_dim = action_dim

    def load_torch(self, filename):
        '''
        Loads trajectories.npz as a flattened torch tensor of (s, a, s')
        '''
        loaded = np.load(filename, allow_pickle=True)
        loaded_data = [loaded[key] for key in loaded]
        states, actions, next_states = loaded_data
        dataset = np.hstack([states, actions, next_states])
        data_tensor = torch.tensor(dataset)
        return data_tensor.to(torch.float32)     
    
    def __getitem__(self, idx):
        '''
        Each row in the .npz trajectories file contains a flattened array of state, action, next_state pairs.
        '''
        # Find the current state by indexing the row 
        x = self.data[idx, :self.state_dim]
        a = self.data[idx, self.state_dim:(self.state_dim + self.action_dim)]
        x_prime = self.data[idx, -self.state_dim:]
        
        return x, a, x_prime
        
    def __len__(self):
        return self.data.shape[0]
    

def load_trajectories_as_torch(filename):
    '''
    Loads trajectories.npz as a flattened torch tensor of (s, a, s')
    '''
    loaded = np.load(filename, allow_pickle=True)
    loaded_data = [loaded[key] for key in loaded]
    states, actions, next_states = loaded_data
    dataset = np.hstack([states, actions, next_states])
    data_tensor = torch.tensor(dataset)
    return data_tensor.to(torch.float32)