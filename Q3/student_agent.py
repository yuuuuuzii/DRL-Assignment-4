import gymnasium as gym
import numpy as np
import torch
from sac import SACAgent
# Do not modify the input of the 'act' function and the '__init__' function. 
def load_checkpoint(agent, filepath: str):

        ckpt = torch.load(filepath, map_location=agent.device)

        agent.actor.load_state_dict(       ckpt['actor'])
        agent.critic.load_state_dict(      ckpt['critic'])
        agent.target_critic.load_state_dict(ckpt['target_critic'])

        agent.actor_optimizer.load_state_dict(  ckpt['actor_optim'])
        agent.critic_optimizer.load_state_dict( ckpt['critic_optim'])

  
        agent.log_alpha.data = ckpt['log_alpha'].to(agent.device)
        agent.alpha_optimizer.load_state_dict( ckpt['alpha_optim'])

        print(f"Loaded checkpoint from {filepath}")

class Agent(object):
    """Agent that acts randomly."""
    def __init__(self):
        self.action_space = gym.spaces.Box(-1.0, 1.0, (21,), np.float64)
        self.agent = SACAgent()
        load_checkpoint(self.agent, 'sac_checkpoint_ep_1048.pt')
    def act(self, observation):
        return self.agent.select_action(observation, True)
