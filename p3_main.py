# General imports
import numpy as np
from collections import deque
from unityagents import UnityEnvironment
import torch

# From current project
from agent import Agent
import utilities as util


def train(env=None, agent=None):
    """Trains a MADDPG-agent on the collaborative Tennis environment.
    :param env: The Reacher unity environment.
    :param agent: An instance of the agent class.
    :return: scores: The training history of achieved scores.
    """
    brain_name = env.brain_names[0]

    scores_deque = deque(maxlen=util.PRINT_EVERY)
    scores = []
    for i_episode in range(1, util.NBR_EPISODES + 1):

        state = env.reset()[brain_name].vector_observations
        state = np.reshape(state, (1, 2*util.OBS_DIMENSION))
        agent.reset()
        done = False
        score = 0
        while not done:
            action = agent.act(state)
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations
            next_state = np.reshape(next_state, (1, 2*util.OBS_DIMENSION))
            reward = np.max(env_info.rewards)
            done = any(env_info.local_done)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
        scores_deque.append(score)
        scores.append(score)
        util.print_results(i_episode, scores_deque, end="")
        if i_episode % util.PRINT_EVERY == 0:
            util.print_results(i_episode, scores_deque)
    return scores

def main():
    # Launch the Tennis unity environment
    env = UnityEnvironment(file_name='Tennis.app')

    util.seeding()

    # Train an agent
    agent = Agent()
    scores = train(env=env, agent=agent)
    util.plot_scores(scores)

    # Save the trained agent to disc
    agent.persist("checkpoint.pth")

    # Reload and test it
    agent_pretrained = Agent()
    agent_pretrained.load("checkpoint.pth")

    N_TEST_EPISODES = 100
    scores = []
    for i_episode in range(1, N_TEST_EPISODES + 1):
        env_info = env.reset(train_mode=True)[env.brain_names[0]]
        state = np.reshape(env_info.vector_observations, (1, 2*util.OBS_DIMENSION))
        score = 0
        while True:
            # action = agent_pretrained.act(state, add_noise=False)

            state = torch.from_numpy(state).float()
            # each agent acts on local states only
            action = []
            with torch.no_grad():
                for a in range(2):
                    agent.actor_local[a].eval()
                    obs_from = a * util.OBS_DIMENSION
                    obs_to = obs_from + util.OBS_DIMENSION
                    action.append(agent.actor_local[a](state[0][obs_from:obs_to]))
                action = torch.cat(action, dim=1).numpy()
                action = np.clip(action, -1, 1)

            env_info = env.step(action)[env.brain_names[0]]
            next_state = np.reshape(env_info.vector_observations, (1, 2*util.OBS_DIMENSION))
            reward = np.max(env_info.rewards)
            done = any(env_info.local_done)
            score += reward
            state = next_state
            if done:
                scores.append(score)
                util.print_results(i_episode, score, end='\r')
                break
    util.print_results(N_TEST_EPISODES, scores)
    util.plot_scores(scores, rolling_window=5)

    env.close()
if __name__ == "__main__":
    main()