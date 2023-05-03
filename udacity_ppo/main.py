from unityagents import UnityEnvironment
import pdb

train_mode = True

env = UnityEnvironment(file_name="../reacher_v2/Reacher.exe")

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

from collections import deque
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim

from ppo_agent_newer_older import Agent

seed_val = 42

from collections import defaultdict

records = defaultdict(list)

## Agent Set-Up
# TRY NOT TO MODIFY: seeding
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


num_steps = 1000
total_timesteps = int(6e6)
num_envs = 20
single_observation_space = 33
single_action_space = 4

anneal_lr = True        #Toggle learning rate annealing for policy and value networks
learning_rate = 3e-4    #the learning rate of the optimizer
gamma = 0.99            #the discount factor gamma
gae_lambda = 0.95       #the lambda for the general advantage estimation
update_epochs = 10      #the K epochs to update the policy
num_minibatches = 32    #the number of mini-batches **Could Change to 4**
clip_coef = 0.2         #the surrogate clipping coefficient
clip_vloss = True       #Toggles whether or not to use a clipped loss for the value function, as per the paper.
norm_adv = True         #Toggles advantages normalization
ent_coef = 0.0          #coefficient of the entropy
vf_coef = 0.5           #coefficient of the value function
max_grad_norm = 0.5     #the maximum norm for the gradient clipping
target_kl = None        #the target KL divergence threshold

batch_size = int(num_envs * num_steps)
minibatch_size = int(batch_size // num_minibatches)

agent = Agent(single_observation_space, single_action_space).to(device)
optimizer = optim.Adam(agent.parameters(), lr=learning_rate, eps=1e-5)

# ALGO Logic: Storage setup
obs = torch.zeros((num_steps, num_envs) + (single_observation_space,)).to(device)
actions = torch.zeros((num_steps,num_envs) + (single_action_space,)).to(device)
logprobs = torch.zeros((num_steps, num_envs)).to(device)
rewards = torch.zeros((num_steps, num_envs)).to(device)
dones = torch.zeros((num_steps, num_envs)).to(device)
values = torch.zeros((num_steps, num_envs)).to(device)

env_info = env.reset(train_mode=train_mode)[brain_name]      # reset the environment    
states = env_info.vector_observations                  # get the current state (for each agent)
num_updates = total_timesteps // batch_size

next_obs = torch.Tensor(states).to(device)
next_done = torch.zeros(num_envs).to(device)
num_updates = total_timesteps // batch_size

global_step = 0
scores_deque = deque(maxlen=2000)

print(num_updates)

for update in range(1, num_updates + 1):

    # Annealing the rate if instructed to do so.
    if anneal_lr:
        frac = 1.0 - (update - 1.0) / num_updates
        lrnow = frac * learning_rate
        optimizer.param_groups[0]["lr"] = lrnow
    u = update - 1
    for s in range(0, num_steps):
        global_step += 1 * num_envs
        obs[s] = next_obs
        dones[s] = next_done

        # ALGO LOGIC: action logic
        with torch.no_grad():
            action, logprob, _, value = agent.get_action_and_value(next_obs)
            values[s] = value.flatten()
        actions[s] = action
        logprobs[s] = logprob
        env_info = env.step(action.cpu().numpy())[brain_name]     # send all actions to tne environment
        next_obs = env_info.vector_observations         # get next state (for each agent)
        reward = env_info.rewards                       # get reward (for each agent)
        done = env_info.local_done                
        rewards[s] = torch.tensor(reward).to(device).view(-1)
        next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

        # Print Score and 
        # Exit loop if any agent is done
        if np.any(done):
            break

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

    
    scores = np.sum(rewards.cpu().numpy(),axis=0)
    for i in scores:
        scores_deque.append(i)
        records['scores'].append(i)

    print('Step: {}\t\tAverage Score: {:.4f}'.format(update,np.mean(scores_deque)))

  
    # flatten the batch
    b_obs = obs.reshape((-1,) + (single_observation_space,))
    b_logprobs = logprobs.reshape(-1)
    b_actions = actions.reshape((-1,) + (single_action_space,))
    b_advantages = advantages.reshape(-1)
    b_returns = returns.reshape(-1)
    b_values = values.reshape(-1)

    # Optimizing the policy and value network
    b_inds = np.arange(batch_size)
    clipfracs = []
    for epoch in range(update_epochs):
        np.random.shuffle(b_inds)
        for start in range(0, batch_size, minibatch_size):
            end = start + minibatch_size
            mb_inds = b_inds[start:end]

            _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
            logratio = newlogprob - b_logprobs[mb_inds]
            ratio = logratio.exp()

            with torch.no_grad():
                # calculate approx_kl http://joschu.net/blog/kl-approx.html
                old_approx_kl = (-logratio).mean()
                approx_kl = ((ratio - 1) - logratio).mean()
                clipfracs += [((ratio - 1.0).abs() > clip_coef).float().mean().item()]

            mb_advantages = b_advantages[mb_inds]
            if norm_adv:
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

            # Policy loss
            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            # Value loss
            newvalue = newvalue.view(-1)
            if clip_vloss:
                v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                v_clipped = b_values[mb_inds] + torch.clamp(
                    newvalue - b_values[mb_inds],
                    -clip_coef,
                    clip_coef,
                )
                v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()
            else:
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

            entropy_loss = entropy.mean()
            loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
            optimizer.step()

        if target_kl is not None:
            if approx_kl > target_kl:
                break

    y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
    var_y = np.var(y_true)
    explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

    # TRY NOT TO MODIFY: record rewards for plotting purposes
    records['learning_rate'].append(optimizer.param_groups[0]["lr"])
    records['value_loss'].append(v_loss.item())
    records['policy_loss'].append(pg_loss.item())
    records['entropy'].append(entropy_loss.item())
    records['old_approx_kl'].append(old_approx_kl.item())
    records['approx_kl'].append(approx_kl.item())
    records['clipfrac'].append(np.mean(clipfracs))
    records['explained_variance'].append(explained_var)

env.close()

torch.save(agent.actor_mean.state_dict(), 'checkpoint_ppo_agent_actor_mean.pth')
torch.save(agent.critic.state_dict(), 'checkpoint_ppo_agent_critic.pth')

import pandas as pd
import matplotlib.pyplot as plt

ppo_logs = pd.DataFrame({'learning_rate':records['learning_rate'],
                         'value_loss':records['value_loss'],
                         'policy_loss':records['policy_loss'],
                         'entropy':records['entropy'],
                         'old_approx_kl':records['old_approx_kl'],
                         'approx_kl':records['approx_kl'],
                         'clipfrac':records['clipfrac'],
                         'explained_variance':records['explained_variance']})
ppo_logs.to_csv('ppo_records_logs.csv')

_scores = pd.DataFrame({'scores':records['scores']})
_scores.to_csv('ddpg_scores_logs.csv')

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(records['learning_rate'])+1), records['learning_rate'])
plt.ylabel('learning_rate')
plt.xlabel('Episode #')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(records['score'])+1), records['score'])
plt.ylabel('score')
plt.xlabel('Episode #')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(records['value_loss'])+1), records['value_loss'])
plt.ylabel('value_loss')
plt.xlabel('Episode #')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(records['policy_loss'])+1), records['policy_loss'])
plt.ylabel('policy_loss')
plt.xlabel('Episode #')
plt.show()