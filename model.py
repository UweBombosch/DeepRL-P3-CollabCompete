import torch
import torch.nn as nn
import torch.nn.functional as F

import utilities as util

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self):
        """Initialize parameters and build model."""
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(util.SEED)
        self.fc1 = nn.Linear(util.OBS_DIMENSION, util.HIDDEN_L1_ACTOR)
        self.fc2 = nn.Linear(util.HIDDEN_L1_ACTOR, util.HIDDEN_L2_ACTOR)
        self.fc3 = nn.Linear(util.HIDDEN_L2_ACTOR, util.ACTION_DIMENSION)
        self.bn1 = nn.BatchNorm1d(util.HIDDEN_L1_ACTOR)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.xavier_normal_(self.fc3.weight)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        if state.dim() == 1:
            state = torch.unsqueeze(state, 0)
        x = F.relu(self.fc1(state))
        x = self.bn1(x)
        x = F.relu(self.fc2(x))
        return F.tanh(self.fc3(x))


class Critic(nn.Module):
    """Critic (Action Value function) model"""

    def __init__(self):
        """Initialize parameters and build model."""
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(util.SEED)

        # The first layer takes in the state to learn a suitable representation of it
        self.fcs1 = nn.Linear(2*util.OBS_DIMENSION, util.HIDDEN_L1_CRITIC)

        # Batch-normalize the output of the first layer, so that the output is not too
        # excessive in comparison to the (clipped) actions
        self.bn = nn.BatchNorm1d(util.HIDDEN_L1_CRITIC)

        # The second layer takes the actions into account, too
        self.fc2 = nn.Linear(util.HIDDEN_L1_CRITIC + 2*util.ACTION_DIMENSION, util.HIDDEN_L2_CRITIC)
        self.fc3 = nn.Linear(util.HIDDEN_L2_CRITIC, 1)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.fcs1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.xavier_normal_(self.fc3.weight)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        if state.dim() == 1:
            state = torch.unsqueeze(state, 0)
        s = F.relu(self.fcs1(state))
        s = self.bn(s)
        # Concatenate the action and the values from previous layer
        x = torch.cat((s, action), dim=1)
        x = F.relu(self.fc2(x))

        return self.fc3(x)