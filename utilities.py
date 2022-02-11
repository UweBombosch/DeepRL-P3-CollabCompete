import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

### Constants

# Parameters for agent learning
BUFFER_SIZE = int(1e+6)
SEED = 29
BATCH_SIZE = 128
NBR_EPISODES = 800
UPDATE_FREQUENCY = 100
GAMMA = 0.99  # discount factor
TAU = .008 # soft update

# Environment parameters
ACTION_DIMENSION = 2
OBS_DIMENSION = 24

# Parameters of the OU-noise process
NOISE_THETA = 0.15
NOISE_SIGMA = 0.2
NOISE_MU = 0.
NOISE_DECAY = 0.99

# Parameters for NNs
HIDDEN_L1_ACTOR = 256
HIDDEN_L2_ACTOR = 128
HIDDEN_L1_CRITIC = 256
HIDDEN_L2_CRITIC = 128
LR_ACTOR = 2e-3  # learning rate of the actor
LR_CRITIC = 2e-3  # learning rate of the critic
WEIGHT_DECAY = 0

PRINT_EVERY = 100


def seeding(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)


def plot_scores(scores, rolling_window=50):
    """Plot scores and optional rolling mean using specified window."""
    plt.figure(figsize=(16, 8))
    plt.plot(scores)
    plt.title("Scores")
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    rolling_mean = pd.Series(scores).rolling(rolling_window).mean()
    plt.plot(rolling_mean)
    plt.show()


def print_results(i_episode, scores, end=None):
    print('\rEpisode {}\tAverage Score: {:.2f}\tStandard Deviation: {:,.2f}'
          .format(i_episode, np.mean(scores), np.std(scores)), end=end)


def soft_update(local_model, target_model):
    """Soft update model parameters.
    θ_target = τ*θ_local + (1 - τ)*θ_target
    Params
    ======
        local_model: PyTorch model (weights will be copied from)
        target_model: PyTorch model (weights will be copied to)
        tau (float): interpolation parameter
    """
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
        target_param.data.copy_(TAU * local_param.data + (1.0 - TAU) * target_param.data)


def hard_copy_weights(source, target):
    """ copy weights from source to target network"""
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

