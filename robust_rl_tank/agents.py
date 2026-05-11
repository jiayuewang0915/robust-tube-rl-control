import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from robust_rl_tank.networks import Actor_error_multi, Actor_model, Critic_error, Critic_model
from robust_rl_tank.replay_buffer import ExperienceReplayBufferError, ExperienceReplayBufferModel


class GaussianActionNoise:
    """Simple decaying Gaussian exploration process."""

    def __init__(self, mean=0, std_decay_rate=0, std_init=np.sqrt(0.1), std_min=0.01, lower_limit=float("-inf"), upper_limit=float("inf")):
        self.mean = mean
        self.std_decay_rate = std_decay_rate
        self.std_init = std_init
        self.std_min = std_min
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit
        self.std_deviation = std_init

    def __call__(self):
        w = self.mean + np.random.normal(size=self.mean.shape) * self.std_deviation
        v = np.clip(w, self.lower_limit, self.upper_limit)
        self.std_deviation = max(self.std_deviation * (1 - self.std_decay_rate), self.std_min)
        return v


class OUActionNoise:
    """Ornstein-Uhlenbeck exploration noise for smooth continuous actions."""

    def __init__(self, mu, Ts=1, sigma=0.2, theta=0.15, dt=1e-2, x0=None):
        self.mu = mu
        self.sigma = sigma
        self.theta = theta
        self.Ts = Ts
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.Ts + self.sigma * np.sqrt(self.Ts) * np.random.normal(size=self.mu.shape)
        self.sigma = max(self.sigma * (1 - self.dt), 0)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)


class DDPGAgentModel:
    """DDPG agent for the nominal spherical tank model."""

    def __init__(
        self,
        env,
        hidden_size=256,
        actor_learning_rate=1e-3,
        critic_learning_rate=1e-3,
        gamma=0.96,
        tau=1e-2,
        max_memory_size=50000,
        file_name=("actor-model.pth", "critic-model.pth"),
    ):
        self.num_state = env.observation_space.shape[0]
        self.num_action = env.action_space.shape[0]
        self.gamma = gamma
        self.tau = tau
        self.file_name = file_name

        # Actor and critic each have slowly updated target networks, following the
        # standard DDPG stabilization strategy.
        self.actor = Actor_model(self.num_state * 2, hidden_size, self.num_action)
        self.actor_target = Actor_model(self.num_state * 2, hidden_size, self.num_action)
        self.critic = Critic_model(self.num_state * 2, self.num_action, hidden_size, self.num_action)
        self.critic_target = Critic_model(self.num_state * 2, self.num_action, hidden_size, self.num_action)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.memory = ExperienceReplayBufferModel(maximum_length=max_memory_size)
        self.critic_criterion = nn.MSELoss()
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_learning_rate)

    def get_action(self, state, error):
        state = torch.from_numpy(state).float().unsqueeze(0)
        error = torch.from_numpy(error).float().unsqueeze(0)
        return self.actor.forward(state, error).detach().numpy()[0, 0]

    def update(self, mini_batch_size):
        state, error, action, reward_model, next_state, next_error, done = self.memory.sample_batch(n=mini_batch_size)
        state = torch.tensor(np.array(state)).float()
        error = torch.tensor(np.array(error)).float()
        action = torch.tensor(np.array(action)).float()
        next_state = torch.tensor(np.array(next_state)).float()
        next_error = torch.tensor(np.array(next_error)).float()
        reward = torch.tensor(np.array(reward_model)).float()
        terminate = torch.tensor(np.array(done)).float().view(-1, 1)

        # Critic target: r + gamma * Q_target(s', actor_target(s')).
        q_values = self.critic.forward(state, error, action.unsqueeze(-1))
        next_action = self.actor_target.forward(next_state, next_error).detach()
        next_q = self.critic_target.forward(next_state, next_error, next_action)
        q_prime = reward + self.gamma * torch.mul(next_q, terminate)
        critic_loss = self.critic_criterion(q_values, q_prime)

        # Deterministic policy gradient objective: choose actions that maximize
        # the critic's value estimate, implemented as minimizing negative Q.
        policy_loss = -self.critic.forward(state, error, self.actor.forward(state, error)).mean()

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self._soft_update(self.actor_target, self.actor)
        self._soft_update(self.critic_target, self.critic)
        return policy_loss.detach().numpy(), critic_loss.detach().numpy()

    def _soft_update(self, target, source):
        """Polyak averaging update for DDPG target networks."""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

    def save(self, models_path):
        os.makedirs(models_path, exist_ok=True)
        torch.save(self.actor, os.path.join(models_path, self.file_name[0]))
        torch.save(self.critic, os.path.join(models_path, self.file_name[1]))


class DDPGAgentError:
    """DDPG agent that learns a compensating action from model-process error."""

    def __init__(
        self,
        env,
        hidden_size=256,
        actor_learning_rate=1e-4,
        critic_learning_rate=1e-3,
        gamma=0.96,
        tau=1e-2,
        max_memory_size=50000,
        file_name=("actor-error.pth", "critic-error.pth"),
    ):
        self.num_state = env.observation_space.shape[0]
        self.num_action = env.action_space.shape[0]
        self.gamma = gamma
        self.tau = tau
        self.file_name = file_name

        # The error agent receives one extra state channel: the real-vs-nominal
        # process mismatch. Its critic therefore estimates the value of correction.
        self.actor = Actor_error_multi(self.num_state * 2, self.num_state, hidden_size, self.num_action)
        self.actor_target = Actor_error_multi(self.num_state * 2, self.num_state, hidden_size, self.num_action)
        self.critic = Critic_error(self.num_state * 2, self.num_state, self.num_action, hidden_size, self.num_action)
        self.critic_target = Critic_error(self.num_state * 2, self.num_state, self.num_action, hidden_size, self.num_action)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.memory = ExperienceReplayBufferError(maximum_length=max_memory_size)
        self.critic_criterion = nn.MSELoss()
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_learning_rate)

    def get_action(self, state, error, state_er):
        state = torch.from_numpy(state).float().unsqueeze(0)
        state_er = torch.from_numpy(state_er).float().unsqueeze(0)
        error = torch.from_numpy(error).float().unsqueeze(0)
        return self.actor.forward(state, error, state_er).detach().numpy()[0, 0]

    def update(self, mini_batch_size):
        state, error, state_er, action_er, reward_er, next_state, next_error, next_state_er, done = self.memory.sample_batch(n=mini_batch_size)
        state = torch.tensor(np.array(state)).float()
        error = torch.tensor(np.array(error)).float()
        state_er = torch.tensor(np.array(state_er)).float()
        action_er = torch.tensor(np.array(action_er)).float()
        next_state = torch.tensor(np.array(next_state)).float()
        next_error = torch.tensor(np.array(next_error)).float()
        next_state_er = torch.tensor(np.array(next_state_er)).float()
        reward = torch.tensor(np.array(reward_er)).float()
        terminate = torch.tensor(np.array(done)).float().view(-1, 1)

        # The error critic evaluates compensation actions conditioned on nominal
        # tracking error and model-process mismatch.
        q_values = self.critic.forward(state, error, state_er, action_er.unsqueeze(-1))
        next_action = self.actor_target.forward(next_state, next_error, next_state_er).detach()
        next_q = self.critic_target.forward(next_state, next_error, next_state_er, next_action)
        q_prime = reward + self.gamma * torch.mul(next_q, terminate)
        critic_loss = self.critic_criterion(q_values, q_prime)

        policy_loss = -self.critic.forward(state, error, state_er, self.actor.forward(state, error, state_er)).mean()

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self._soft_update(self.actor_target, self.actor)
        self._soft_update(self.critic_target, self.critic)
        return policy_loss.detach().numpy(), critic_loss.detach().numpy()

    def _soft_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

    def save(self, models_path):
        os.makedirs(models_path, exist_ok=True)
        torch.save(self.actor, os.path.join(models_path, self.file_name[0]))
        torch.save(self.critic, os.path.join(models_path, self.file_name[1]))


DDPGAgent_model = DDPGAgentModel
DDPGAgent_error = DDPGAgentError
