import torch
import numpy as np
import pandas as pd
from collections import deque
import matplotlib.pyplot as plt

from ddpg_agent import Agent
from unityagents import UnityEnvironment

from easydict import EasyDict as e_dict

import pdb


seed = 42

train_mode = True

env = UnityEnvironment(file_name=r"C:\\Users\\micha\\OneDrive\\Data Science\Udacity\\Deep Reinforcement Learning\\continous-control-robot-arm\\reacher_v2\\Reacher.exe")

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=train_mode)[brain_name]

# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space 
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])

ddpg_config = e_dict()

ddpg_config.state_size = 33         # enviroment size
ddpg_config.action_size = 4         # action size
ddpg_config.seed = 42

ddpg_config.update_step = 20

## DDPG Params
ddpg_config.buffer_size = int(5e5)  # replay buffer size
ddpg_config.batch_size = 256        # minibatch size
ddpg_config.gamma = 0.99            # discount factor
ddpg_config.tau = 1e-3              # for soft update of target parameters
ddpg_config.lr_actor = 1e-4         # learning rate of the actor
ddpg_config.lr_critic = 1e-3        # learning rate of the critic
ddpg_config.weight_decay = 0.0     # L2 weight decay
ddpg_config.loss = 'l1_smooth'      # loss functions include 'mae' or 'mse'

## Noise Params
ddpg_config.theta = 0.15
ddpg_config.sigma = 0.2
ddpg_config.add_noise = True

agent = Agent(ddpg_config)

n_episodes=1000
max_t=1000
print_every=20

scores_deque = deque(maxlen=2000)
scores_eps_deque = deque(maxlen=20)
scores = []
eps_actor_loss = []
eps_critic_loss = []
for i_episode in range(1, n_episodes+1):
    env_info = env.reset(train_mode=train_mode)[brain_name] # reset the environment
    state = env_info.vector_observations                 # get the current state
    agent.reset()
    score = np.zeros((20,))
    for t in range(max_t):
        
        action = np.apply_along_axis(lambda x: agent.act(x,ddpg_config.add_noise), axis=1, arr=state)
        
        env_info = env.step(action)[brain_name]         # send the action to the environment
        next_state = env_info.vector_observations       # get the next state
        reward = env_info.rewards                       # get the reward
        done = env_info.local_done                      # see if episode has finished
        
        for i in range(20):
            a_loss, c_loss = agent.step(state[i], action[i], reward[i], next_state[i], done[i], t)
        if a_loss: 
            eps_actor_loss.append(a_loss)
        if c_loss:
            eps_critic_loss.append(c_loss)

        state = next_state
        score += np.array(reward)
        if np.any(done):
            break
    if i_episode == 250:
        ddpg_config.add_noise = False
    
    # pdb.set_trace()
    for i in range(20):
        scores_deque.append(score[i])
        scores_eps_deque.append(score[i])
        scores.append(score[i])

    if i_episode % print_every == 0:
        print('\rEpisode {}\tAverage Score: {:.4f}\tEpisode Score: {:.4f}\tCritic Loss: {:.5f}\tActor Loss: {:.5f}'.format(i_episode, np.mean(scores_deque),  np.mean(scores_eps_deque), eps_critic_loss[-1], eps_actor_loss[-1]))
        torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
        torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
    elif eps_critic_loss:
        print('\rEpisode {}\tAverage Score: {:.4f}\tEpisode Score: {:.4f}\tCritic Loss: {:.5f}\tActor Loss: {:.5f}'.format(i_episode, np.mean(scores_deque),  np.mean(scores_eps_deque), eps_critic_loss[-1], eps_actor_loss[-1]),end="")

#pdb.set_trace()

logs = pd.DataFrame({'scores':scores,'actor_loss':eps_actor_loss, 'critic_loss':eps_critic_loss})
logs.to_csv('logs.csv')

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores)+1), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(eps_actor_loss)+1), eps_actor_loss)
plt.ylabel('Loss')
plt.xlabel('Episode #')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(eps_critic_loss)+1), eps_critic_loss)
plt.ylabel('Loss')
plt.xlabel('Episode #')
plt.show()

env.close()