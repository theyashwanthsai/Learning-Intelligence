# Grid World Q-Learning: Design Document

## 🎯 What We're Building

A visual simulation where an AI agent learns to navigate a 5x5 grid maze to reach a goal while avoiding obstacles - **completely from scratch**.

## 🎮 How It Will Look

```
Visual Grid (before training):
S . . . .
. X . . .
. . X . .
. X . . .
. . . . G

Visual Grid (after training):
S→→→→.
. X ↓ . ↓
. . X→→↓
. X . . ↓
. . . .→G

Legend: S=Start, G=Goal, X=Obstacle, →↓=Learned path
```

## 🏗️ Architecture

### 1. **Environment (GridWorld Class)**
- **Purpose**: The "game world" that defines rules
- **Components**:
  - 5x5 grid with positions
  - Agent starting at (0,0)
  - Goal at (4,4) 
  - 3 obstacles as barriers
- **Functions**: Move agent, calculate rewards, check if done

### 2. **Agent (QLearningAgent Class)**
- **Purpose**: The "brain" that learns optimal moves
- **Components**:
  - Q-table: 25×4 matrix (25 positions × 4 actions)
  - Learning parameters (learning rate, exploration rate)
- **Functions**: Choose actions, update knowledge from experience

### 3. **Training Loop**
- **Purpose**: Let agent practice thousands of times
- **Process**: Reset → Explore → Learn → Repeat

## 🔄 How It Works

### Learning Process:
1. **Agent starts randomly moving** (high exploration)
2. **Gets rewards**: +100 for goal, -10 for obstacles, -1 for each step
3. **Updates Q-table**: "This action from this position was good/bad"
4. **Gradually becomes smarter** (reduces exploration)
5. **Eventually finds optimal path**

### Q-Learning Magic:
- **Q-table stores "action values"** for each position
- **Higher Q-value = better action**
- **Agent learns**: "From position X, action Y leads to best long-term reward"

## 📊 Expected Results

**Episode 1-100**: Random wandering, lots of failures
**Episode 100-500**: Starts finding goal occasionally  
**Episode 500-1000**: Consistently finds optimal path
**Final**: Agent takes shortest route every time

## 🛠️ Implementation Plan

1. **GridWorld class** (environment rules)
2. **QLearningAgent class** (learning algorithm)  
3. **Training loop** (practice sessions)
4. **Visualization** (see the learning happen)
