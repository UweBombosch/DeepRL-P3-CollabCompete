[//]: # (Image References)
[train]: ./images/Training-Performance.png
[test]: ./images/Test-Performance.png
[Lowe_Wu]: ./images/Lowe_Wu.png
# Solution Report P3 Collaboration and Competition

## Overview
Our implementation solves the `Tennis` environment with  a variant of the **multi-agent deep deterministic policy gradient (MADDPG)** algorithm that has 
been described by Ryan Lowe and Yi Wu in 
[[Lowe & Wu 2020]](https://arxiv.org/abs/1706.02275).
It solves the environment in less than 600 episodes with a score higher than 0.5 as required, but attains a six times 
better value at about 800 episodes before plateauing.


## The Algorithm in a Nutshell

There are environments that involve the interaction between multiple agents, where the emergent action policies 
arise from agents co-evolving together. Some of these environments can by construction not be solved by a single 
central intelligence, for instance  because they imply a competition of agents with conflicting goal settings, or 
because the information about states and actions that would have to be centralized exceeds available bandwidths. 
Other environments can in principle be tackled by just a single agent brain, but better solutions are achieved 
by a collaboration of several widely independent actors.

Lowe and Wu propose a general framework for multi-agent learning that is based on centralized training with 
decentralized execution. It is an actor-critic approach, but with several actors and critics. While it is a 
mandate that the actor only receives local information in order to be able to act at execution time in a 
decentralized way, the critic may use centralized information at training time that encompasses the perspectives 
of several clients. Note that the critic is only needed at training, not at execution time. 
The framework is illustrated by Figure 1 in
[[Lowe & Wu 2020]](https://arxiv.org/abs/1706.02275):

![Lowe_Wu]
 
In the very general setting, MADDPG consists of an arbitrary number N of agent-critic pairs. The i-th actor (i = 1,...,N)
learns an own policy &pi;<sub>i</sub> that just receives local observations o<sub>i</sub> of the global 
environment state. In contrast to this, each critic's state-value function Q<sub>i</sub> has access to all local 
observations (o<sub>1</sub>, ..., o<sub>N</sub>) and actions (a<sub>1</sub>, ..., a<sub>N</sub>). It therefore
criticizes its actor in learning with an awareness of what all the others are doing and achieving.

When working on our solution, we experimented with variants of this general setting. One degenerate case is
DDPG, where there only is a single actor-critic pair. This is too simplistic, and showed scores of about 0.2 in best
case. The other extreme is a full implementation of MADDPG with two independent actors and critics. This turned
out to be unstable - it learns quite well for some episodes, but then apparently forgets everything, or is
highly depending on parameters like SEED. But since both of our tennis-actors have a common goal - keeping the ball
in the game - we turned to a collaborative setting, where there are two independent actors, but only one critic.  

Both, actors and critic are implemented by artificial neural networks in `pytorch`.

Like in the earlier `P2 Reacher` project, we also include the following ingredients:

1. We use an **Ornstein-Uhlenbeck** stochastic process to add a little noise to the actions of the actors in order to 
ensure that the algorithm keeps exploring sufficiently long.
2. We use a **Replay Buffer**  to de-correlate TD(0)-frames in training.
3. We use **off-policy learning** by introducing twins of actor and critic in order to avoid unstable learning. 
More specifically, actor and critic both come in two copies that are identical except for the weights: 
    1. A **behavior network** that is not learning by gradient ascent, but dictates the behavior and represents the policy 
    (a.k.a. target network). 
    2. A **learning network** that is frequently updated by gradient ascent with batches from the Replay Buffer 
    (a.k.a local network).
    
After each learning step, the algorithm mixes a percentage of the weights of the learning networks into the weights
of the behavior networks according to the scheme: w<sub>b</sub> = &tau; w<sub>l</sub> + (1-&tau;) w<sub>b</sub>, 
where 0 &le; &tau; &le; 1.

## Architecture and Hyperparameters

Each **actor** is a straightforward fully connected NN with the following settings:

| <!-- -->  | <!-- --> |
| --- | --- |
| Number of hidden layers | 2|
| Nodes per hidden layer | (256, 128) | 
| Input layer | 24 float-values, namely 8 observations of the racket and ball position for three consecutive times.|
| Output layer | 2 float-values in the range of -1 to 1, determining the forward-backward, and up-down movement of 
the racket. | 

The **critic** is a trifle more complicated. Its input consists of two parts, a state s and an action a. Following 
the code-examples of the Udemy-course, we enter the state in the input layer, but the action is added only to the
second layer "from the side". We tried the more natural alternative to enter the full tuple (s, a) already at the
input layer, but the result was worse.

| <!-- -->  | <!-- -->  |
| --- | --- |
| Number of hidden layers | 2|
| Nodes per hidden layer | (256 + 4 action values, 128) | 
| Input layer | 48 + 4 float-values consisting of the 2 x 24 agent observations and 2 x 2 actions |
| Output layer | Single float number representing the Q-value | 

Except for the output layers, both ANNs use the `ReLu` activation function. In learning, we use the Adam-optimizer 
with default settings. Weights are initialized using the **Xavier Normal** scheme, which by experiments gave the 
best results.

Other settings that we used are:

| <!-- -->  | <!-- -->  |
| --- | --- |
| Replay Buffer size | 1,000,000|
| Batch size for backpropagation | 128 |
| Learning rate | 0.002 |
| Discount factor &gamma; | 0.99 |
| Soft update rate &tau; | 0.008 |
| Noise decay | 0.99 |
| Weight decay | None |

All parameters are available as constants in the `utilities.py` module. We extensively experimented with other 
settings, but this is the optimum we observed.

## Results
Here is a report about the **training performance**:

![Training Scores][train]


After training, we persist the agent to disk, and reload it in order to **assess its performance on 100 consecutive 
episodes**. Here is a typical result:

![Test Scores][test]

The average of scores over 100 episodes in this case is 3.62. We executed this test several times, receiving similar 
results with a variance of about +/- 0.8.

## Criticism 
Training and test performance show a high variance. There are episodes, where the agents collaborate in an almost
perfect way, but others where they completely fail. When looking at the agents acting, their play looks somewhat
stereotypic and monotone. If the initial position of the ball suits them well, they ping-pong endlessly - otherwise, 
they fail at once. This can be due to the single critic that synchronizes the players. A second critic may have
prepared the player for imperfect returns of the co-player. 



 