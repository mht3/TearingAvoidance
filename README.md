# Tearing Avoidance Control

Forked from [TearingAvoidance](https://github.com/PlasmaControl/TearingAvoidance)

- Tearing mode instability is the leading cause of plasma disruption in a tokamak.
- This repository provides the Python scripts for training the ML models:\
  (1) Tearing mode prediction model using deep learning\
  (2) Tearing mode avoidance model using deep reinforcement learning

# Note
- Some scripts need experimental data from DIII-D, which are not available from this repository.   To access DIII-D data, you should become a DIII-D user, per the instructions at https://d3dfusion.org/become-a-user/.

# Environment Setup
First, clone the repository.
```
git clone https://github.com/mht3/TearingAvoidance
cd TearingAvoidance
```
# Conda Environment

Create the fusion_rl environment with Python 3.8
```
conda create -n fusion_rl python=3.8
```

Activate the environment
```
conda activate fusion_rl
```

Install the required packages
```
pip install -r requirements.txt
```

## Install Torch

Choose between cuda and cpu versions.

### Option 1: CPU Only

```
pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cpu
```
### Option 2: Cuda 12.1

```
pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu121
```
## Install TensorFlow

Choose between cuda and cpu versions.

### Option 1: CPU Only

```
pip install tensorflow==2.4.1 keras-rl2
```

# References
- J. Seo et al., "Avoiding fusion plasma tearing instability with deep reinforcement learning." Nature [626 (2024): 746.](https://www.nature.com/articles/s41586-024-07024-9)
