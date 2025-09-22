import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.adamw import AdamW
from torch.distributions import MultivariateNormal, Categorical
import tqdm
from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        n_hidden_layers: int,
    ):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_hidden_layers = n_hidden_layers

        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)

        self.fcnm1 = nn.ModuleList([nn.Linear(self.hidden_dim, self.hidden_dim) for _ in range(self.n_hidden_layers)])
        self.fcn = nn.Linear(self.hidden_dim, self.output_dim)

        self.F = nn.ReLU()
        self.finalF = nn.Identity()

    def forward(self, x: torch.Tensor):
        x = self.F(self.fc1(x))
        for layer in self.fcnm1:
            x = self.F(layer(x))
        x = self.finalF(self.fcn(x))
        return x


class MLPK(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        keysize: int,
        n_hidden_layers: int,
    ):
        super(MLPK, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_hidden_layers = n_hidden_layers
        self.keysize = keysize
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)

        self.fcnm1 = nn.ModuleList([nn.Linear(self.hidden_dim, self.hidden_dim) for _ in range(self.n_hidden_layers)])
        self.fcn = nn.Linear(self.hidden_dim, self.output_dim)

        self.kc1 = nn.Linear(self.keysize, self.hidden_dim)
        self.kcnm1 = nn.ModuleList([nn.Linear(self.hidden_dim, self.hidden_dim) for _ in range(self.n_hidden_layers)])
        self.kcn = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.F = nn.ReLU()
        self.finalF = nn.Identity()
        self.KF = nn.ReLU()
        self.finalKF = nn.Tanh()

    def forward(self, x: torch.Tensor, k=None):
        if k is None:
            x = self.F(self.fc1(x))
            for layer in self.fcnm1:
                x = self.F(layer(x))
            x = self.finalF(self.fcn(x))
            return x
        else:
            z = self.KF(self.kc1(k))
            for layer in self.kcnm1:
                z = self.KF(layer(z))
            z = self.finalKF(self.kcn(z))
            x = self.F(self.fc1(x))
            for layer in self.fcnm1:
                x = self.F(z * layer(x))
            x = self.finalF(self.fcn(x))
            return x


class MNISTClassifier(nn.Module):
    def __init__(self, keysize=32):
        super(MNISTClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 2, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 2, 1)
        self.conv3 = nn.Conv2d(32, 64, 3, 2, 1)
        self.conv4 = nn.Conv2d(64, 128, 3, 2, 1)

        self.fc1 = nn.Linear(512, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 10)

        self.k1 = nn.Linear(keysize, 64)
        self.k2 = nn.Linear(64, 64)
        self.k3 = nn.Linear(64, 64)

        self.bn2 = nn.BatchNorm2d(32, 0.8)
        self.bn3 = nn.BatchNorm2d(64, 0.8)
        self.bn4 = nn.BatchNorm2d(128, 0.8)

        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x: torch.Tensor, k=None):
        x = self.dropout(self.relu(self.conv1(x)))
        x = self.bn2(self.dropout(self.relu(self.conv2(x))))
        x = self.bn3(self.dropout(self.relu(self.conv3(x))))
        x = self.bn4(self.dropout(self.relu(self.conv4(x))))
        x = x.reshape(x.shape[0], -1)
        if k is None:
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            x = self.sigmoid(self.fc3(x))
            return x
        else:
            z = self.tanh(self.k1(k))
            z = self.tanh(self.k2(z))
            z = self.tanh(self.k3(z))
            x = self.relu(self.fc1(x))
            x = self.relu(z * self.fc2(x))
            x = self.sigmoid(self.fc3(x))
            return x

    def sample(self, x, k=None):
        x = self.forward(x, k)
        x = torch.multinomial(x / torch.sum(x), 1)
        return x


class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
        self.keys = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]
        del self.keys[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init, device=None):
        super(ActorCritic, self).__init__()

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(self.device)
        # actor
        if has_continuous_action_space:
            self.actor = nn.Sequential(nn.Linear(state_dim, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, action_dim))
        else:
            self.actor = nn.Sequential(nn.Linear(state_dim, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, action_dim), nn.Softmax(dim=-1))
        # critic
        self.critic = nn.Sequential(nn.Linear(state_dim, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 1))

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(self.device)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def forward(self):
        raise NotImplementedError

    def act(self, state, eval=False):

        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(state)

        if eval and self.has_continuous_action_space:
            return action_mean.detach(), action_logprob.detach(), state_val.detach()  # type: ignore
        else:
            return action.detach(), action_logprob.detach(), state_val.detach()

    def evaluate(self, state, action):

        if self.has_continuous_action_space:
            action_mean = self.actor(state)

            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(self.device)
            dist = MultivariateNormal(action_mean, cov_mat)

            # For Single Action Environments.
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy


class PPO:
    """this class was ripped from a random repo i found on github. no license, so fair game"""

    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std_init=0.6, device=None):

        self.has_continuous_action_space = has_continuous_action_space
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        if has_continuous_action_space:
            self.action_std = action_std_init

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init, device=device).to(self.device)
        self.optimizer = torch.optim.Adam(
            [{"params": self.policy.actor.parameters(), "lr": lr_actor}, {"params": self.policy.critic.parameters(), "lr": lr_critic}]
        )

        self.policy_old = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init, device=device).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling PPO::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        print("--------------------------------------------------------------------------------------------")
        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if self.action_std <= min_action_std:
                self.action_std = min_action_std
                print("setting actor output action_std to min_action_std : ", self.action_std)
            else:
                print("setting actor output action_std to : ", self.action_std)
            self.set_action_std(self.action_std)

        else:
            print("WARNING : Calling PPO::decay_action_std() on discrete action space policy")
        print("--------------------------------------------------------------------------------------------")

    def select_action(self, state, eval=False):

        if self.has_continuous_action_space:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(self.device)
                action, action_logprob, state_val = self.policy_old.act(state, eval=eval)

            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_val)

            return action.detach().cpu().numpy().flatten()
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(self.device)
                action, action_logprob, state_val = self.policy_old.act(state, eval=eval)

            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_val)

            return action.item()

    def update(self):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(self.device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(self.device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(self.device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(self.device)

        # calculate advantages
        advantages = rewards.detach() - old_state_values.detach()
        # Optimize policy for K epochs
        for _ in range(self.K_epochs):

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))


class ActorCriticK(nn.Module):
    def __init__(self, state_dim, action_dim, keysize, has_continuous_action_space, action_std_init, device=None):
        super(ActorCriticK, self).__init__()

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(self.device)
        # actor
        self.actor1 = nn.Linear(state_dim, 128)
        self.actor2 = nn.Linear(128, 128)
        self.actor3 = nn.Linear(128, 128)
        self.actor4 = nn.Linear(128, action_dim)

        self.critic1 = nn.Linear(state_dim, 128)
        self.critic2 = nn.Linear(128, 128)
        self.critic3 = nn.Linear(128, 128)
        self.critic4 = nn.Linear(128, 1)

        self.key_encoder1 = nn.Sequential(nn.Linear(keysize, 64), nn.ReLU(), nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 128), nn.Tanh())
        self.key_encoder2 = nn.Sequential(nn.Linear(keysize, 64), nn.ReLU(), nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 128), nn.Tanh())

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(self.device)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def forward(self):
        raise NotImplementedError

    def act(self, state, key=None, eval=False):

        if self.has_continuous_action_space:
            x = F.relu(self.actor1(state))
            x = F.relu(self.actor2(x))
            if key is not None:
                z = self.key_encoder1(key)
                x = F.relu(self.actor3(x) * z)
            else:
                x = F.relu(self.actor3(x))
            action_mean = self.actor4(x)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            x = F.relu(self.actor1(state))
            x = F.relu(self.actor2(x))
            if key is not None:
                z = self.key_encoder1(key)
                x = F.relu(self.actor3(x) * z)
            else:
                x = F.relu(self.actor3(x))
            action_probs = F.softmax(self.actor4(x), dim=-1)
            dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)

        v = F.relu(self.critic1(state))
        v = F.relu(self.critic2(v))
        if key is not None:
            z = self.key_encoder2(key)
            v = F.relu(self.critic3(v) * z)
        else:
            v = F.relu(self.critic3(v))
        v = self.critic4(v)

        if eval and self.has_continuous_action_space:
            return action_mean.detach(), action_logprob.detach(), v.detach()  # type: ignore
        else:
            return action.detach(), action_logprob.detach(), v.detach()

    def evaluate(self, state, action, key=None):

        if self.has_continuous_action_space:
            x = F.relu(self.actor1(state))
            x = F.relu(self.actor2(x))
            if key is not None:
                z = self.key_encoder1(key)
                x = F.relu(self.actor3(x) * z)
            else:
                x = F.relu(self.actor3(x))
            action_mean = self.actor4(x)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)
        else:
            x = F.relu(self.actor1(state))
            x = F.relu(self.actor2(x))
            if key is not None:
                z = self.key_encoder1(key)
                x = F.relu(self.actor3(x) * z)
            else:
                x = F.relu(self.actor3(x))
            action_probs = F.softmax(self.actor4(x), dim=-1)
            dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        v = F.relu(self.critic1(state))
        v = F.relu(self.critic2(v))
        if key is not None:
            z = self.key_encoder2(key)
            v = F.relu(self.critic3(v) * z)
        else:
            v = F.relu(self.critic3(v))
        state_values = self.critic4(v)

        return action_logprobs, state_values, dist_entropy


class PPOK:
    """this class was ripped from a random repo i found on github. no license, so fair game"""

    def __init__(self, state_dim, action_dim, keysize, lr, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std_init=0.6, device=None):

        self.has_continuous_action_space = has_continuous_action_space
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        if has_continuous_action_space:
            self.action_std = action_std_init

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.buffer = RolloutBuffer()

        self.policy = ActorCriticK(state_dim, action_dim, keysize, has_continuous_action_space, action_std_init, device=device).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr)

        self.policy_old = ActorCriticK(state_dim, action_dim, keysize, has_continuous_action_space, action_std_init, device=device).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling PPO::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        print("--------------------------------------------------------------------------------------------")
        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if self.action_std <= min_action_std:
                self.action_std = min_action_std
                print("setting actor output action_std to min_action_std : ", self.action_std)
            else:
                print("setting actor output action_std to : ", self.action_std)
            self.set_action_std(self.action_std)

        else:
            print("WARNING : Calling PPO::decay_action_std() on discrete action space policy")
        print("--------------------------------------------------------------------------------------------")

    def select_action(self, state, key=None, eval=False):

        if self.has_continuous_action_space:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(self.device)
                action, action_logprob, state_val = self.policy_old.act(state, key, eval=eval)

            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_val)
            self.buffer.keys.append(key)

            return action.detach().cpu().numpy().flatten()
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(self.device)
                action, action_logprob, state_val = self.policy_old.act(state, key, eval=eval)

            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            self.buffer.state_values.append(state_val)
            self.buffer.keys.append(key)

            return action.item()

    def update(self):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(self.device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(self.device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(self.device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(self.device)
        keys = torch.squeeze(torch.stack(self.buffer.keys, dim=0)).detach().to(self.device)

        # calculate advantages
        advantages = rewards.detach() - old_state_values.detach()
        # Optimize policy for K epochs
        for _ in range(self.K_epochs):

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions, keys)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))


class CVAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, cond_dim: int, output_dim: int):
        super(CVAE, self).__init__()
        self.enc1 = nn.Linear(input_dim, input_dim)
        self.enc2 = nn.Linear(input_dim, input_dim)
        self.enc3 = nn.Linear(input_dim, latent_dim)
        self.mean = nn.Linear(latent_dim, latent_dim)
        self.std = nn.Linear(latent_dim, latent_dim)

        self.dec1 = nn.Linear(latent_dim + cond_dim, latent_dim)
        self.dec2 = nn.Linear(latent_dim, latent_dim * 2)
        self.dec3 = nn.Linear(latent_dim * 2, output_dim)
        
        # giving it a ton of layers so that it stands a chance

    def forward(self, x, k):
        mu, logstd = self.encode(x)
        logstd = torch.clamp(logstd, -20.0, 2.0)
        std = torch.exp(0.5 * logstd)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return self.decode(z, k)

    def encode(self, x):
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        x = F.relu(self.enc3(x))
        m = self.mean(x)
        s = self.std(x)
        return m, s

    def decode(self, z, k):
        x = torch.concat((z, k), dim=1)
        x = F.relu(self.dec1(x))
        x = F.relu(self.dec2(x))
        x = self.dec3(x)
        return x

    def kld(self, x):
        mu, logstd = self.encode(x)
        return -0.5 * torch.sum(1 + logstd - mu.pow(2) - logstd.exp())