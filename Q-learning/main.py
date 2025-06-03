from train import train_agent, visualize_learning, test_agent
from environment import GridWorld
from q_learning import QLearningAgent
import numpy as np

# Train the agent
env, agent, rewards, steps = train_agent(1000)

# Visualize learning progress
visualize_learning(rewards, steps)

# Test the trained agent
test_agent(env, agent)

# Show final Q-table (optional)
print("\nFinal Q-table shape:", agent.q_table.shape)
print("Max Q-values per state:", np.max(agent.q_table, axis=1))