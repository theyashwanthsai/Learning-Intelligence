from train import train_agent_with_visualization, create_episode_timelapse_mp4, test_agent
from environment import GridWorld
from q_learning import QLearningAgent
import numpy as np

# Train the agent
env, agent, rewards, steps, all_paths = train_agent_with_visualization(episodes=1000)

create_episode_timelapse_mp4(all_paths, speed_multiplier=4)

