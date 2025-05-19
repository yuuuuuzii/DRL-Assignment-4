
from util import Actor, Critic, ReplayBuffer
import torch.optim as optim
import torch
import torch.nn as nn
import gym
import numpy as np
import ipdb
import torch
from torch import nn, optim

class SACAgent:
    def __init__(self, device=None):
        self.state_dim  = 67
        self.action_dim = 21
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor         = Actor(self.state_dim, self.action_dim).to(self.device)
        self.critic        = Critic(self.state_dim, self.action_dim).to(self.device)
        self.target_critic = Critic(self.state_dim, self.action_dim).to(self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.actor_optimizer  = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=3e-4)
      
        self.target_entropy = -float(self.action_dim)

        self.log_alpha = nn.Parameter(torch.tensor(-1.609, dtype=torch.float32, device=self.device))
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=3e-4)

        self.buffer = ReplayBuffer()
        self.gamma  = 0.99
        self.tau    = 0.005

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, state, deterministic):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        with torch.no_grad():
            if deterministic:
                mean, _ = self.actor(state)
                action = torch.tanh(mean)
            else:
                action, _ = self.actor.sample(state)
        action = action[0].cpu().numpy()
        return action

    def train(self, batch_size=128):
        if len(self.buffer) < batch_size:
            return
        
        state, action, reward, next_state, done = self.buffer.sample(batch_size)

        alpha = self.alpha
        # Critic loss
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_state)
            target_q1, target_q2 = self.target_critic(next_state, next_action)

            target_value = reward + (1 - done) * self.gamma * (torch.min(target_q1, target_q2) - alpha * next_log_prob)

        q1 ,q2 = self.critic(state, action)
        critic_loss = nn.MSELoss()(q1, target_value)+ nn.MSELoss()(q2, target_value)
        
      
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        ## Actor loss
        action_pi, log_prob_pi = self.actor.sample(state)
        q1_pi ,q2_pi = self.critic(state, action_pi)
  
        min_q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (alpha * log_prob_pi - min_q_pi).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.alpha_optimizer.zero_grad()
        alpha_loss = -(self.log_alpha * (log_prob_pi + self.target_entropy).detach()).mean()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        ##soft
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
