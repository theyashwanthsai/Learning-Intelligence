import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
from environment import GridWorld
from q_learning import QLearningAgent
import matplotlib.patches as patches

def train_agent_with_visualization(episodes=1000):
    """Modified training function that captures episode paths for visualization"""
    # Create environment and agent
    env = GridWorld()
    agent = QLearningAgent()

    # Track progress AND paths
    episode_rewards = []
    episode_steps = []
    all_episode_paths = []  # NEW: Store all episode paths

    print("Training started...")

    for episode in range(episodes):
        # Reset environment
        state = env.reset()
        total_reward = 0
        steps = 0
        episode_path = [env.agent_pos.copy()]  # NEW: Track agent path

        # Run one episode
        while True:
            # Agent chooses action
            action = agent.choose_action(state)

            # Environment responds
            next_state, reward, done = env.step(action)
            episode_path.append(env.agent_pos.copy())  # NEW: Record position

            # Agent learns from experience
            agent.update_q_table(state, action, reward, next_state)

            # Update for next iteration
            state = next_state
            total_reward += reward
            steps += 1

            # End episode if goal reached or too many steps
            if done or steps > 100:
                break

        # Store episode data
        all_episode_paths.append(episode_path)  # NEW: Store path
        agent.decay_epsilon()
        episode_rewards.append(total_reward)
        episode_steps.append(steps)

        # Print progress every 100 episodes
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_steps = np.mean(episode_steps[-100:])
            print(f"Episode {episode+1}: Avg Reward = {avg_reward:.1f}, Avg Steps = {avg_steps:.1f}, Epsilon = {agent.epsilon:.3f}")

    return env, agent, episode_rewards, episode_steps, all_episode_paths

def create_episode_timelapse_mp4(all_episode_paths, grid_size=5, obstacles=None, save_mp4=True, speed_multiplier=1, filename='q_learning_episodes.mp4'):
    """
    Create an animated timelapse of all Q-Learning episodes and save as MP4
    
    Args:
        all_episode_paths: List of episode paths from training
        grid_size: Size of the grid
        obstacles: List of (x,y) obstacle positions  
        save_mp4: Whether to save as MP4
        speed_multiplier: How fast to play (1=normal, 2=2x speed, etc.)
        filename: Output MP4 filename
    """
    
    if obstacles is None:
        obstacles = [(1, 1), (2, 2), (3, 1)]  # Default grid obstacles
    
    # Setup figure - use Agg backend to avoid display
    plt.ioff()  # Turn off interactive mode
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Colors: empty, obstacle, start, goal, agent, trail
    colors = ['white', 'black', 'green', 'red', 'blue', 'orange']
    cmap = ListedColormap(colors)
    
    # Create base grid
    base_grid = np.zeros((grid_size, grid_size))
    for ox, oy in obstacles:
        base_grid[oy, ox] = 1
    base_grid[0, 0] = 2  # Start
    base_grid[4, 4] = 3  # Goal
    
    # Left plot: Episode animation
    im1 = ax1.imshow(base_grid, cmap=cmap, vmin=0, vmax=5)
    ax1.set_title('Q-Learning Episodes Timelapse', fontsize=14, fontweight='bold')
    ax1.set_xticks(range(grid_size))
    ax1.set_yticks(range(grid_size))
    ax1.grid(True, alpha=0.3)
    
    # Right plot: Learning curve
    episode_lengths = [len(path) for path in all_episode_paths]
    ax2.set_xlim(0, len(all_episode_paths))
    ax2.set_ylim(0, max(episode_lengths) + 5)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Steps to Goal')
    ax2.set_title('Learning Progress', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Progress tracking
    progress_line, = ax2.plot([], [], 'b-', alpha=0.7, linewidth=2)
    current_point, = ax2.plot([], [], 'ro', markersize=8)
    
    # Text displays
    episode_text = ax1.text(0.02, 0.98, '', transform=ax1.transAxes, 
                           fontsize=12, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
    
    stats_text = ax2.text(0.02, 0.98, '', transform=ax2.transAxes,
                         fontsize=10, verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9))
    
    # Store arrow patches for reuse
    arrow_patches = []
    
    def draw_path_arrows(path):
        """Draw arrows showing the path direction"""
        # Clear previous arrows
        for patch in arrow_patches:
            patch.remove()
        arrow_patches.clear()
        
        # Draw arrows between consecutive positions
        for i in range(len(path) - 1):
            x1, y1 = path[i]
            x2, y2 = path[i + 1]
            
            # Calculate arrow direction
            dx = x2 - x1
            dy = y2 - y1
            
            if dx != 0 or dy != 0:  # Only draw if there's movement
                # Arrow from center of current cell to center of next cell
                arrow = patches.FancyArrowPatch(
                    (x1, y1), (x2, y2),
                    arrowstyle='->', 
                    mutation_scale=15,
                    color='purple',
                    alpha=0.8,
                    linewidth=2
                )
                ax1.add_patch(arrow)
                arrow_patches.append(arrow)
    
    def animate(frame):
        episode_idx = frame // (10 // speed_multiplier)
        
        if episode_idx >= len(all_episode_paths):
            return im1, progress_line, current_point, episode_text, stats_text
        
        current_path = all_episode_paths[episode_idx]
        
        # Update grid
        current_grid = base_grid.copy()
        
        # Show full path for current episode
        for i, (x, y) in enumerate(current_path):
            if i == len(current_path) - 1:
                current_grid[y, x] = 4  # Agent (blue)
            elif i > 0:  # Don't overwrite start
                current_grid[y, x] = 5  # Trail (orange)
        
        im1.set_array(current_grid)
        
        # Draw path arrows
        draw_path_arrows(current_path)
        
        # Update episode info
        success_status = "SUCCESS" if current_path[-1] == [4, 4] else "FAILED"
        episode_text.set_text(f'Episode: {episode_idx + 1}/{len(all_episode_paths)}\n'
                             f'Steps: {len(current_path)}\n'
                             f'Status: {success_status}')
        
        # Update progress chart
        episodes_so_far = list(range(1, episode_idx + 2))
        steps_so_far = episode_lengths[:episode_idx + 1]
        
        progress_line.set_data(episodes_so_far, steps_so_far)
        current_point.set_data([episode_idx + 1], [len(current_path)])
        
        # Update stats
        if episode_idx >= 9:  # After 10 episodes
            recent_avg = np.mean(episode_lengths[max(0, episode_idx-9):episode_idx+1])
            best_so_far = min(episode_lengths[:episode_idx+1])
            success_rate = sum(1 for path in all_episode_paths[:episode_idx+1] 
                             if path[-1] == [4, 4]) / (episode_idx + 1) * 100
            
            stats_text.set_text(f'Recent Avg: {recent_avg:.1f} steps\n'
                               f'Best: {best_so_far} steps\n'
                               f'Success Rate: {success_rate:.1f}%')
        
        return im1, progress_line, current_point, episode_text, stats_text
    
    # Create animation
    total_frames = len(all_episode_paths) * (10 // speed_multiplier)
    interval = max(50 // speed_multiplier, 10)  # Minimum 10ms interval
    
    anim = animation.FuncAnimation(fig, animate, frames=total_frames, 
                                 interval=interval, blit=False, repeat=False)
    
    plt.tight_layout()
    
    if save_mp4:
        print(f"Saving MP4 as '{filename}'... This may take a few minutes for {len(all_episode_paths)} episodes...")
        
        # Save as MP4 using ffmpeg writer
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=20, metadata=dict(artist='Q-Learning Visualization'), bitrate=1800)
        
        anim.save(filename, writer=writer)
        print(f"MP4 saved as '{filename}'")
    
    plt.close(fig)  # Close figure to free memory
    return anim

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



    
# Create timelapse (choose one):

# Option 1: Fast timelapse (4x speed)
# create_episode_timelapse_mp4(all_paths, speed_multiplier=4)

# Option 2: Show every 10th episode for quicker viewing
# every_10th = all_paths[::10]
# create_episode_timelapse(every_10th, save_gif=True)

# Option 3: Show just first 100 episodes in detail
# first_100 = all_paths[:100]
# create_episode_timelapse(first_100, speed_multiplier=2, save_gif=True)