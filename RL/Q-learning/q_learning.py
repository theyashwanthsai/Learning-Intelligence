import numpy as np
import matplotlib.pyplot as plt
import random

class QLearningAgent:
    def __init__(self, states=25, actions=4):
        # Q-table: rows=states(25), columns=actions(4)
        self.q_table = np.zeros((states, actions))

        # Learning parameters
        self.learning_rate = 0.1    # How fast to learn (alpha)
        self.discount_factor = 0.9  # How much to value future rewards (gamma)
        self.epsilon = 1.0          # Exploration rate (start with 100% random)
        self.epsilon_decay = 0.995  # Reduce exploration over time
        self.epsilon_min = 0.01     # Minimum exploration (always keep 1%)

    def choose_action(self, state):
        """
        Choose action using epsilon-greedy strategy:
        - With probability epsilon: explore (random action)
        - With probability 1-epsilon: exploit (best known action)
        """
        if random.random() < self.epsilon:
            # Explore: choose random action
            return random.randint(0, 3)
        else:
            # Exploit: choose best action from Q-table
            return np.argmax(self.q_table[state])

    def update_q_table(self, state, action, reward, next_state):
        """
        Update Q-table using Q-learning formula:
        Q(s,a) = Q(s,a) + α[r + γ*max(Q(s',a')) - Q(s,a)]
        """
        # Current Q-value
        current_q = self.q_table[state, action]

        # Best future Q-value
        max_future_q = np.max(self.q_table[next_state])

        # Q-learning update
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_future_q - current_q
        )

        # Update Q-table
        self.q_table[state, action] = new_q

    def decay_epsilon(self):
        """Gradually reduce exploration as agent learns"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay