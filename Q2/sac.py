
from util import Actor, Critic, ReplayBuffer
import torch.optim as optim
import torch
import torch.nn as nn
import gym
import numpy as np
class SACAgent:
    def __init__(self):
    
        self.state_dim = 5
        self.action_dim = 1
        
        # 取得 Action 的最大值
        self.max_action = 1.0

        self.actor = Actor(self.state_dim, self.action_dim, self.max_action)
        self.critic_1 = Critic(self.state_dim, self.action_dim)
        self.critic_2 = Critic(self.state_dim, self.action_dim)
        self.target_critic_1 = Critic(self.state_dim, self.action_dim)
        self.target_critic_2 = Critic(self.state_dim, self.action_dim)

        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=3e-4)
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=3e-4)

        self.buffer = ReplayBuffer()
        self.gamma = 1
        self.tau = 0.005
        self.alpha = 0.2

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor(state).detach().cpu().numpy()[0]
        return action

    def train(self, batch_size=128):
        if len(self.buffer) < batch_size:
            return
        
        state, action, reward, next_state, done = self.buffer.sample(batch_size)

        # Critic loss
        with torch.no_grad():
            next_action = self.actor(next_state)
            target_q1 = self.target_critic_1(next_state, next_action)
            target_q2 = self.target_critic_2(next_state, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_action
            target_value = reward + (1 - done) * self.gamma * target_q

        q1 = self.critic_1(state, action)
        q2 = self.critic_2(state, action)

        critic_1_loss = nn.MSELoss()(q1, target_value)
        critic_2_loss = nn.MSELoss()(q2, target_value)

        # Update critics
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()

        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        # Actor loss
        actor_loss = -self.critic_1(state, self.actor(state)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update target networks
        for target_param, param in zip(self.target_critic_1.parameters(), self.critic_1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for target_param, param in zip(self.target_critic_2.parameters(), self.critic_2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
