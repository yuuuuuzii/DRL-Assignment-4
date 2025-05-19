from util import ReplayBuffer
from sac import SACAgent
import gymnasium as gym
import ipdb
import os
import torch

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


env = gym.make("Pendulum-v1")
agent = SACAgent()

for episode in range(200):
    state,_= env.reset()
    episode_reward = 0

    for step in range(300):
        action = agent.select_action(state)
        next_state, reward, done, truncated, _ = env.step(action)
        agent.buffer.push(state, action, reward, next_state, done)

        agent.train()
        state = next_state
        episode_reward += reward

        if done or truncated:
            break
    
    print(f"Episode {episode}, Reward: {episode_reward}")
    
save_checkpoint(agent, 199)

