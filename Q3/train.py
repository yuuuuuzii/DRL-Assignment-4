
from sac import SACAgent
import gymnasium as gym

import os
import torch
from dmc import make_dmc_env
import numpy as np

def make_env():
	# Create environment with state observations
	env_name = "humanoid-walk"
	env = make_dmc_env(env_name, np.random.randint(0, 1000000), flatten=True, use_pixels=False)
	return env

def save_checkpoint(agent, episode, path="checkpoints"):
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save({
            'actor':            agent.actor.state_dict(),
            'critic':           agent.critic.state_dict(),
            'target_critic':    agent.target_critic.state_dict(),
            # optimizers
            'actor_optim':      agent.actor_optimizer.state_dict(),
            'critic_optim':     agent.critic_optimizer.state_dict(),
            # alpha 自動調整參數
            'log_alpha':        agent.log_alpha.detach().cpu(),
            'alpha_optim':      agent.alpha_optimizer.state_dict(),
    }, f"{path}/sac_checkpoint_ep_{episode}.pt")
    print(f"Checkpoint saved at episode {episode}")

env = make_env()
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

agent = SACAgent()
num_episodes = 10000


for episode in range(num_episodes):
    state, _ = env.reset()
    episode_reward = 0
    done = False
    while not done:

        action = agent.select_action(state, False)
             
        next_state, reward, done, truncated, _ = env.step(action)
        agent.buffer.push(state, action, reward, next_state, done)

        agent.train()
        state = next_state
        episode_reward += reward
        if done or truncated:
            break
    if (episode+1) % 50 == 0:
        save_checkpoint(agent, episode-1)
    print(f"Episode {episode}, Reward: {episode_reward:.2f}")
save_checkpoint(agent, num_episodes-1)
