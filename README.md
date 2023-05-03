# Continous Control of Robotic  Arm
**Michael Santoro - micheal.santoro@du.edu**
# Introduction
This project explores two different Reinforcement Algorithims. This particular enviroment requires continious control. The two algorithims that will be explored are [Deep Deterministic Policy Gradient](https://arxiv.org/abs/1509.02971) (DDPG) [Proximal Policy Opimization](https://arxiv.org/abs/1707.06347) (PPO). DDPG and PPO are two popular algorithms used for solving continuous control problems in reinforcement learning.

DDPG is a model-free, off-policy algorithm that is an extension of the actor-critic architecture. It is designed to handle continuous action spaces, which can be difficult for other algorithms like DQN. DDPG uses two neural networks: an actor network that maps states to actions, and a critic network that estimates the Q-value of a given state-action pair. The actor network is trained to maximize the Q-value estimated by the critic network. DDPG has been shown to be effective in solving a variety of continuous control problems, such as robotic arm control and autonomous driving.

PPO, on the other hand, is a model-free, on-policy algorithm that is designed to handle both discrete and continuous action spaces. It uses a trust region optimization approach to update the policy, which ensures that the new policy is not too far from the old policy. This helps to prevent the policy from diverging during training. PPO has been shown to be effective in solving a variety of continuous control problems, such as humanoid locomotion and robotic manipulation.

So, why might you want to use DDPG and PPO for continuous control problems? Well, both algorithms have been shown to be effective in solving a variety of continuous control problems, and they each have their own strengths and weaknesses. DDPG is good for problems where the action space is continuous and the environment is deterministic, while PPO is good for problems where the action space is both discrete and continuous and the environment is stochastic. Ultimately, the choice between DDPG and PPO (or any other algorithm) will depend on the specific problem you are trying to solve and the characteristics of the environment you are working with.

This a solution submission for the Deep Reinforcement Learning by Udacity. The problem details can be found [here](https://github.com/udacity/deep-reinforcement-learning/tree/master/p2_continuous-control).

# Enviroment
In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

There were two unity enviroments provided. The first version contains a single agent. The second version contains 20 identical agents, each with its own copy of the environment.

The environment is considered solved, when the average (over 100 episodes) of those average scores is at least +30.

# Enviroment Set-Up
1. Find the appropiate unity reacher enviroment from this [repo](https://github.com/udacity/deep-reinforcement-learning/tree/master/p2_continuous-control).
2. After cloning the repository navigate to the repo and run the following command.
```
conda env create --file drlnd.yml
```
3. Run the Following.
```
conda activate drlnd
```

