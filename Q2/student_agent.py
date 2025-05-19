import gymnasium
import numpy as np
from sac import SACAgent
import torch

def load_checkpoint(agent, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    agent.actor.load_state_dict(checkpoint['actor_state_dict'])
    agent.critic_1.load_state_dict(checkpoint['critic_1_state_dict'])
    agent.critic_2.load_state_dict(checkpoint['critic_2_state_dict'])
    agent.target_critic_1.load_state_dict(checkpoint['target_critic_1_state_dict'])
    agent.target_critic_2.load_state_dict(checkpoint['target_critic_2_state_dict'])
    agent.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
    agent.critic_1_optimizer.load_state_dict(checkpoint['critic_1_optimizer_state_dict'])
    agent.critic_2_optimizer.load_state_dict(checkpoint['critic_2_optimizer_state_dict'])
    print(f"Checkpoint loaded from {checkpoint_path}")

# Do not modify the input of the 'act' function and the '__init__' function. 
class Agent(object):
    """Agent that acts randomly."""
    def __init__(self):
        self.action_space = gymnasium.spaces.Box(-1.0, 1.0, (1,), np.float64)
        self.agent = SACAgent()
        load_checkpoint(self.agent, 'checkpoints/sac_checkpoint_ep_119.pt')
    def act(self, observation):
        return self.agent.select_action(observation)
