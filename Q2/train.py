
from sac import SACAgent
import gymnasium as gym
import ipdb
import os
import torch
from dmc import make_dmc_env
import numpy as np

def make_env():
	# Create environment with state observations
	env_name = "cartpole-balance"
	env = make_dmc_env(env_name, np.random.randint(0, 1000000), flatten=True, use_pixels=False)
	return env

def save_checkpoint(agent, episode, path="checkpoints"):
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save({
        'actor_state_dict': agent.actor.state_dict(),
        'critic_1_state_dict': agent.critic_1.state_dict(),
        'critic_2_state_dict': agent.critic_2.state_dict(),
        'target_critic_1_state_dict': agent.target_critic_1.state_dict(),
        'target_critic_2_state_dict': agent.target_critic_2.state_dict(),
        'actor_optimizer_state_dict': agent.actor_optimizer.state_dict(),
        'critic_1_optimizer_state_dict': agent.critic_1_optimizer.state_dict(),
        'critic_2_optimizer_state_dict': agent.critic_2_optimizer.state_dict(),
    }, f"{path}/sac_checkpoint_ep_{episode}.pt")
    print(f"Checkpoint saved at episode {episode}")

env = make_env()
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

agent = SACAgent()
num_episodes = 120
max_steps = 1100
rewards = []
for episode in range(num_episodes):
    state, _ = env.reset()
    episode_reward = 0
    for step in range(max_steps):
        action = agent.select_action(state)
        next_state, reward, done, truncated, _ = env.step(action)
        agent.buffer.push(state, action, reward, next_state, done)
        agent.train()
        state = next_state
        episode_reward += reward
        if done or truncated:
            break
    if(np.mean(rewards[-5:])> 970):
        save_checkpoint(agent, episode-1)
        break
    print(f"Episode {episode}, Reward: {episode_reward:.2f}")

save_checkpoint(agent, num_episodes-1)