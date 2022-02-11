import numpy as np
import random
import copy
from collections import namedtuple, deque
import torch
import torch.nn.functional as F
import torch.optim as optim

from model import Actor, Critic
import utilities as util

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent:
    """Interacts with and learns from the environment."""

    def __init__(self):
        """Initialize an Agent object.
        """

        random_seed = util.SEED
        random.seed(random_seed)

        # Actor Network (w/ Target Network)
        self.actor_local = []
        self.actor_target = []
        self.actor_optimizer = []
        for a in range(2):
            self.actor_local.append(Actor().to(device))
            self.actor_target.append(Actor().to(device))
            self.actor_optimizer.append(optim.Adam(self.actor_local[a].parameters(),
                                                   lr=util.LR_ACTOR, weight_decay=util.WEIGHT_DECAY))

        # Critic Network (w/ Target Network)
        self.critic_local = Critic().to(device)
        self.critic_target = Critic().to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(),
                                           lr=util.LR_CRITIC, weight_decay=util.WEIGHT_DECAY)

        for a in range(2):
            util.hard_copy_weights(self.actor_local[a], self.actor_target[a])
            util.hard_copy_weights(self.actor_local[a], self.actor_target[a])
        util.hard_copy_weights(self.critic_local, self.critic_target)

        # Noise process
        self.noise = OUNoise()
        self.noise_decay = util.NOISE_DECAY

        # Replay memory
        self.memory = ReplayBuffer()

    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > util.BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences)

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)

        action = []
        for a in range(2):
            self.actor_local[a].eval()
            with torch.no_grad():
                obs_from = a*util.OBS_DIMENSION
                obs_to = obs_from + util.OBS_DIMENSION
                action.append(self.actor_local[a](state[0][obs_from:obs_to]))
            self.actor_local[a].train()

        action = torch.cat(action, dim=1).numpy()

        if add_noise:
            action += self.noise_decay*self.noise.sample()
            self.noise_decay *= self.noise_decay
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = torch.cat([self.actor_target[0](next_states[:, :util.OBS_DIMENSION]),
                                  self.actor_target[1](next_states[:, util.OBS_DIMENSION:])], dim=1)
        Q_targets_next = self.critic_target(next_states, actions_next)

        # Compute Q targets for current states (y)
        Q_targets = rewards + (util.GAMMA * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = []
        actions_pred.append(self.actor_local[0](states[:, :util.OBS_DIMENSION]))
        actions_pred.append(self.actor_local[1](states[:, util.OBS_DIMENSION:]))
        actions_pred = torch.cat(actions_pred, dim=1)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        for a in range(2):
            self.actor_optimizer[a].zero_grad()
        actor_loss.backward()
        for a in range(2):
            self.actor_optimizer[a].step()

        # ----------------------- update target networks ----------------------- #
        util.soft_update(self.critic_local, self.critic_target)
        for a in range(2):
            util.soft_update(self.actor_local[a], self.actor_target[a])

    def persist(self, filename):
        """Persists the weight of the actor and critic neural networks to a file
        Params
        ======
            filename (string): Fully qualified name of the file.
        """
        state_dict = {'actor': [self.actor_target[0].state_dict(), self.actor_target[1].state_dict()],
                      'critic': self.critic_target.state_dict()}
        torch.save(state_dict, filename)

    def load(self, filename):
        """Loads the weights of the actor and critic neural network weights from a file.
        Params
        ======
            filename (string): Fully qualified name of the file.
        """
        state_dict = torch.load(filename)
        self.__init__()
        for a in range(2):
            self.actor_local[a].load_state_dict(state_dict['actor'][a])
            util.hard_copy_weights(self.actor_local[a], self.actor_target[a])
        self.critic_local.load_state_dict(state_dict['critic'])
        util.hard_copy_weights(self.critic_local, self.critic_target)

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self):
        """Initialize parameters and noise process."""
        self.mu = util.NOISE_MU * np.ones((2*util.ACTION_DIMENSION))
        self.theta = util.NOISE_THETA
        self.sigma = util.NOISE_SIGMA
        random.seed(util.SEED)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = util.ACTION_DIMENSION
        self.memory = deque(maxlen=util.BUFFER_SIZE)  # internal memory (deque)
        self.batch_size = util.BATCH_SIZE
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = util.SEED
        random.seed(self.seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)