[//]: # (Image References)
[tennis]: ./images/tennis.gif
# Submission to Project P3 Collaboration and Competition

This repository contains a solution of the **P3 Collaboration and Competiton** challenge of the 
[Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893).

## Project Details
Udacity describes the task to be solved as follows
[[Ref]](https://classroom.udacity.com/nanodegrees/nd893/parts/f0b328e5-de4f-4a4e-a788-a9965fc2692a/modules/020946b2-198c-45df-8d59-75cd66c0713a/lessons/66375d73-93e2-4be9-83e8-c9a5432a1c1e/concepts/4f073f75-5547-43ab-9359-f86427362b0e):
 
In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, 
it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a 
reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

![tennis][tennis]

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. 
Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement 
toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 
(over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

* After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. 
This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
* This yields a single score for each episode.

The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.

We will solve the task with a variant of the **multi-agent deep deterministic policy gradient (MADDPG)** algorithm that has 
been described by Ryan Lowe and Yi Wu in 
[[Lowe & Wu 2020]](https://arxiv.org/abs/1706.02275).
The solution outperforms the required benchmark of +0.5 by a factor of 6. 


## Getting Started

The solution has been implemented with the Python 3.6.13 distribution of Anaconda 3. We will use a virtual environment
`p3venv` with packages and versions as specified in the `requirements.txt` file.

The Unity Tennis environment for MacOS has been downloaded as requested in Step 2 of 
[Section 4. The Environment - Explore](https://classroom.udacity.com/nanodegrees/nd893/parts/abb587d8-60cc-4d3f-a628-8f0af120c94a/modules/d08bc8d7-fdfb-42a1-9fe4-62f5d8dcfff2/lessons/5da2debd-eae0-4a70-b21f-be1603870c27/concepts/dc754a0c-d5e1-4a04-98dc-b16bd1a93371).

The GitHub repository contains the following files:
* `agent.py`: Python-implementation of the MADDPG-agent. Includes the replay-buffer and the Ornstein-Uhlenbeck process.
* `model.py`: Python class that implements the actor and the critic fully connected neural networks.
* `utilities.py`: Assembles all parameters as global constants and static functions for plotting, printing, or copying 
weights of neural networks.
* `p3_main.py`: Main process for training the agent.
* `CollabCompete.ipynb`: Jupyter notebook for training and testing the MADDPG-agent.
* `checkpoint.pth`: A persisted version of the agent, including all trained neural networks, weights and parameters.
* `setup.py` and `requirements.txt`: Configuration files for installing packages automatically with `pip install`

## Instructions

### Installation on Linux or MacOS
Note that the setup of the environment is exactly the same as the `drlnd` virtual environment that is used throughout
the nanodegree training.

Install Anaconda 4.9.0 (or higher) according to [these instructions](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).
Create and activate a virtual environment `p3venv` for testing the solution in isolation as follows:

```shell script
conda create --name p3venv python=3.6
conda activate p3venv
```
Clone the repository of this solution (or alternatively unzip the zip-file) into a directory of your choice, and
install the needed libraries.

```shell script
git clone https://github.com/UweBombosch/DeepRL-P3-CollabCompete.git
cd DeepRL-P3-CollabCompete
pip install .
```

Ceate a kernel for Jupyter:
````shell script
python -m ipykernel install --user --name p3venv --display-name "p3venv"
````

Install the ``Tennis`` environment as described [here](https://classroom.udacity.com/nanodegrees/nd893/parts/abb587d8-60cc-4d3f-a628-8f0af120c94a/modules/d08bc8d7-fdfb-42a1-9fe4-62f5d8dcfff2/lessons/5da2debd-eae0-4a70-b21f-be1603870c27/concepts/dc754a0c-d5e1-4a04-98dc-b16bd1a93371).
On a Mac, the OS might prohibit the ``Tennis.app`` to start, because the app was not downloaded from the
Apple Store. In this case, open the System Settings, go to Security, open the lock, and grant permissions to
``Tennis.app``.

Start Jupyter, open the notebook ``CollabCompete.ipynb``, and make 
sure it uses the ``p3venv`` kernel. The agent can be trained and tested by walking through the
notebook - please just follow the instructions provided there.





 

