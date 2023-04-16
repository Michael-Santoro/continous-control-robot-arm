from collections import namedtuple, deque
import numpy as np
import random
import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DummyBody(nn.Module):
    def __init__(self, state_dim):
        super(DummyBody, self).__init__()
        self.feature_dim = state_dim

    def forward(self, x):
        return x

def layer_init(layer, w_scale=1.0):
    nn.init.orthogonal_(layer.weight.data)
    layer.weight.data.mul_(w_scale)
    nn.init.constant_(layer.bias.data, 0)
    return layer
    
class DeterministicActorCriticNet(nn.Module):
    def __init__(self,
                 state_dim,
                 action_dim,
                 actor_opt_fn,
                 critic_opt_fn,
                 phi_body=None,
                 actor_body=None,
                 critic_body=None):
        super(DeterministicActorCriticNet, self).__init__()
        if phi_body is None: phi_body = DummyBody(state_dim)
        if actor_body is None: actor_body = DummyBody(phi_body.feature_dim)
        if critic_body is None: critic_body = DummyBody(phi_body.feature_dim)
        self.phi_body = phi_body
        self.actor_body = actor_body
        self.critic_body = critic_body
        self.fc_action = layer_init(nn.Linear(actor_body.feature_dim, action_dim), 1e-3)
        self.fc_critic = layer_init(nn.Linear(critic_body.feature_dim, 1), 1e-3)

        self.actor_params = list(self.actor_body.parameters()) + list(self.fc_action.parameters())
        self.critic_params = list(self.critic_body.parameters()) + list(self.fc_critic.parameters())
        self.phi_params = list(self.phi_body.parameters())
        
        self.actor_opt = actor_opt_fn(self.actor_params + self.phi_params)
        self.critic_opt = critic_opt_fn(self.critic_params + self.phi_params)
        self.to(device)

    def forward(self, obs):
        phi = self.feature(obs)
        action = self.actor(phi)
        return action

    def feature(self, obs):
        obs = tensor(obs)
        return self.phi_body(obs)

    def actor(self, phi):
        return torch.tanh(self.fc_action(self.actor_body(phi)))

    def critic(self, phi, a):
        return self.fc_critic(self.critic_body(torch.cat([phi, a], dim=1)))

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class RandomProcess(object):
    def reset_states(self):
        pass

class LinearSchedule:
    def __init__(self, start, end=None, steps=None):
        if end is None:
            end = start
            steps = 1
        self.inc = (end - start) / float(steps)
        self.current = start
        self.end = end
        if end > start:
            self.bound = min
        else:
            self.bound = max

    def __call__(self, steps=1):
        val = self.current
        self.current = self.bound(self.current + self.inc * steps, self.end)
        return val
    

class OrnsteinUhlenbeckProcess(RandomProcess):
    def __init__(self, size, std, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = 0
        self.std = std
        self.dt = dt
        self.x0 = x0
        self.size = size
        self.reset_states()

    def sample(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.std() * np.sqrt(
            self.dt) * np.random.randn(*self.size)
        self.x_prev = x
        return x

    def reset_states(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros(self.size)

def tensor(x):
    if isinstance(x, torch.Tensor):
        return x
    x = np.asarray(x, dtype=np.float32)
    x = torch.from_numpy(x).to(device)
    return x

class DDPGAgent():
    def __init__(self, config):
        self.config = config
        # Randomly Initialize the Critic Network
        self.network = DeterministicActorCriticNet(config.state_dim, 
                                                   config.action_dim,
                                                   config.actor_opt_fn,
                                                   config.critic_opt_fn)
        
        # Randomly Initialize the target Network
        self.target_network = DeterministicActorCriticNet(config.state_dim, 
                                                   config.action_dim,
                                                   config.actor_opt_fn,
                                                   config.critic_opt_fn)
        self.target_network.load_state_dict(self.network.state_dict())
        
        # Expose these Params to Notebook?
        self.replay = ReplayBuffer(action_size=config.action_dim, buffer_size=int(1e6), batch_size=100, seed=config.seed)
        self.random_process = OrnsteinUhlenbeckProcess(size=(config.action_dim,), std=LinearSchedule(0.2))
        self.total_steps = 0
        self.state = None

    def soft_update(self, target, src):
        for target_param, param in zip(target.parameters(), src.parameters()):
            target_param.detach_()
            target_param.copy_(target_param * (1.0 - self.config.target_network_mix) +
                               param * self.config.target_network_mix)

    def eval_step(self, state):
        self.config.state_normalizer.set_read_only()
        state = self.config.state_normalizer(state)
        action = self.network(state)
        self.config.state_normalizer.unset_read_only()
        return to_np(action)

    def step(self, state, action, reward, next_state, done):
        config = self.config

        self.replay.add(state,action,reward,next_state,done)

        if done:
            self.random_process.reset_states()
        self.state = next_state
        self.total_steps += 1

        if len(self.replay) >= config.warm_up:
            transitions = self.replay.sample()
            states = tensor(transitions[0])
            actions = tensor(transitions[1])
            rewards = tensor(transitions[2]).unsqueeze(-1)
            next_states = tensor(transitions[3])

            print(next_states.shape)
            phi_next = self.target_network.feature(next_states)
            print(phi_next.shape)
            a_next = self.target_network.actor(phi_next)
            print(a_next.shape)
            q_next = self.target_network.critic(phi_next, a_next)
            q_next = config.discount * q_next
            q_next.add_(rewards)
            q_next = q_next.detach()
            phi = self.network.feature(states)
            q = self.network.critic(phi, actions)
            critic_loss = (q - q_next).pow(2).mul(0.5).sum(-1).mean()

            self.network.zero_grad()
            critic_loss.backward()
            self.network.critic_opt.step()

            phi = self.network.feature(states)
            action = self.network.actor(phi)
            policy_loss = -self.network.critic(phi.detach(), action).mean()

            self.network.zero_grad()
            policy_loss.backward()
            self.network.actor_opt.step()

            self.soft_update(self.target_network, self.network)
            
        
    def act(self, state):
        config = self.config
        num_agents=1 
        if self.total_steps < config.warm_up:
            action = np.random.randn(num_agents, config.action_dim)
        else:
            action = self.network(self.state)
            action = to_np(action)
            action += self.random_process.sample()
        action = np.clip(action, -1,1)
        return action
