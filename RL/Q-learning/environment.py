import numpy as np
import matplotlib.pyplot as plt
import random

class GridWorld:
    def __init__(self):
        self.size = 5
        self.agent_pos = [0, 0]  # Start position (top-left)
        self.goal_pos = [4, 4]   # Goal position (bottom-right)
        self.obstacles = [[1,1], [2,2], [3,1]]  # Obstacle positions

    def reset(self):
        """Reset agent to starting position"""
        self.agent_pos = [0, 0]
        return self.get_state()

    def get_state(self):
        """Convert 2D position to 1D state number (0-24)"""
        return self.agent_pos[0] * self.size + self.agent_pos[1]
    
    def step(self, action):
        """
        Take action and return (new_state, reward, done)
        Actions: 0=up, 1=right, 2=down, 3=left
        """
        # Store old position
        old_pos = self.agent_pos.copy()

        # Move based on action
        if action == 0:  # Up
            self.agent_pos[0] = max(0, self.agent_pos[0] - 1)
        elif action == 1:  # Right
            self.agent_pos[1] = min(self.size-1, self.agent_pos[1] + 1)
        elif action == 2:  # Down
            self.agent_pos[0] = min(self.size-1, self.agent_pos[0] + 1)
        elif action == 3:  # Left
            self.agent_pos[1] = max(0, self.agent_pos[1] - 1)

        # Calculate reward
        reward = self.get_reward()

        # Check if episode is done
        done = (self.agent_pos == self.goal_pos)

        # Print grid
        grid = [['.' for _ in range(self.size)] for _ in range(self.size)]
        grid[self.agent_pos[0]][self.agent_pos[1]] = 'A'
        grid[self.goal_pos[0]][self.goal_pos[1]] = 'G'
        for obs in self.obstacles:
            grid[obs[0]][obs[1]] = 'X'
        
        # print('\nGrid:')
        for row in grid:
            # print(' '.join(row))
            pass

        return self.get_state(), reward, done
    
    def get_reward(self):
        """Calculate reward for current position"""
        if self.agent_pos == self.goal_pos:
            return 100  # Big reward for reaching goal
        elif self.agent_pos in self.obstacles:
            return -10  # Penalty for hitting obstacle
        else:
            return -1   # Small penalty for each step (encourages efficiency)
        
def grid_test():
    env = GridWorld()
    print("Reset state: ", env.reset())
    print("Action 1: ", env.step(1))
    print("Action 3: ", env.step(3))
    print("Action 0: ", env.step(0))
    print("Action 1: ", env.step(1))
    print("Action 2: ", env.step(2))
    print("Reward: ", env.get_reward())

# grid_test()