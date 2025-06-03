import numpy as np
import matplotlib.pyplot as plt
from environment import GridWorld
from q_learning import QLearningAgent

def train_agent(episodes=1000):
    # Create environment and agent
    env = GridWorld()
    agent = QLearningAgent()

    # Track progress
    episode_rewards = []
    episode_steps = []

    print("Training started...")

    for episode in range(episodes):
        # Reset environment
        state = env.reset()
        total_reward = 0
        steps = 0

        # Run one episode
        while True:
            # Agent chooses action
            action = agent.choose_action(state)

            # Environment responds
            next_state, reward, done = env.step(action)

            # Agent learns from experience
            agent.update_q_table(state, action, reward, next_state)

            # Update for next iteration
            state = next_state
            total_reward += reward
            steps += 1

            # End episode if goal reached or too many steps
            if done or steps > 100:
                break

        # Reduce exploration
        agent.decay_epsilon()

        # Track progress
        episode_rewards.append(total_reward)
        episode_steps.append(steps)

        # Print progress every 100 episodes
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_steps = np.mean(episode_steps[-100:])
            print(f"Episode {episode+1}: Avg Reward = {avg_reward:.1f}, Avg Steps = {avg_steps:.1f}, Epsilon = {agent.epsilon:.3f}")

    return env, agent, episode_rewards, episode_steps

def visualize_learning(episode_rewards, episode_steps):
    """Plot learning progress"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Plot rewards
    ax1.plot(episode_rewards)
    ax1.set_title('Episode Rewards Over Time')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')

    # Plot steps to goal
    ax2.plot(episode_steps)
    ax2.set_title('Steps to Goal Over Time')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Steps')

    plt.tight_layout()
    plt.show()

def test_agent(env, agent, num_tests=5):
    """Test the trained agent"""
    print("\nTesting trained agent:")

    for test in range(num_tests):
        state = env.reset()
        path = [env.agent_pos.copy()]
        steps = 0

        while True:
            # Use best action (no exploration)
            action = np.argmax(agent.q_table[state])
            state, reward, done = env.step(action)
            path.append(env.agent_pos.copy())
            steps += 1

            if done or steps > 20:
                break

        print(f"Test {test+1}: {steps} steps, Path: {path}")



