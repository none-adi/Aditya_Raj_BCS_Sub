from random import randint
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, deque



'''
---------------------------------------------------------SUPPORT FUNCTIONS-------------------------------------------------------------
'''
def load_maze(filename):
    maze = []
    with open(filename, "r") as file:
        for line in file:
            maze.append(list(line.strip("\n")))
    return maze

def reset(maze):
    while True:
        harry_x, harry_y = randint(0,9), randint(0,14)
        death_x, death_y = randint(0,9), randint(0,14)
        cup_x, cup_y = randint(0,9), randint(0,14)
        harry, death, cup = (harry_x, harry_y), (death_x, death_y), (cup_x, cup_y)
        if harry != death and harry != cup and death != cup and maze[harry_x][harry_y] != 1 and maze[death_x][ death_y] != 1 and maze[cup_x][ cup_y] != 1:
            return (harry_x,harry_y, death_x,death_y,cup_x,cup_y)
        
# updates the Q table
def q_update(Q,state,action,reward,next_state,alpha,gamma):
    Q[state][action] = Q[state][action] + alpha*(reward+gamma*max(Q[next_state].values()) - Q[state][action])
    return Q

# Gives the next action of harry using epsilon greedy
def harry_action(Q, state, epsilon):
    harry_state_current = [state[0], state[1]]
    if random.random() < epsilon:
        # Explore: choose random valid action
        actions = [0, 1, 2, 3]
        random.shuffle(actions)
        for action in actions:
            next_harry_state = is_action_valid(harry_state_current, action)
            if next_harry_state:
                return next_harry_state, action
    else:
        # Exploit: choose best valid action based on Q-values
        # Sorts the dictionary as (action, q_value) pairs in the reverse order
        q_actions = sorted(Q[state].items(), key=lambda x: x[1], reverse=True)
          
        for action, _ in q_actions:
            next_harry_state = is_action_valid(harry_state_current, action)
            if next_harry_state:
                return next_harry_state,action

    # In case no valid action found (shouldn't happen)
    return harry_state_current

# Updates the state based on the new moves
def update_state(state, harry_new_state, death_new_state):
    # state has 6 elements, while harry and death new state have 2
    HX,HY,DX,DY,CX,CY = state
    hx,hy = harry_new_state
    dx, dy = death_new_state
    return (hx,hy,dx,dy,CX,CY)

# Implementation of Breadth First Algorithm
# Function to get valid neighbors from a given cell (up, down, left, right)
def get_neighbors(pos, maze, cup_pos):
    neighbors = []
    x, y = pos
    rows = len(maze)
    cols = len(maze[0])

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right

    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        # Check if the new position is within bounds and not a wall (assuming wall = 1)
        if 0 <= nx < rows and 0 <= ny < cols and maze[nx][ny] != 1 and (nx,ny) != cup_pos:
            neighbors.append((nx, ny))

    return neighbors

# BFS function to get the next move for Death Eater to reach Harry
def death_action(maze, start, goal, cup_pos):
    """
    maze: 2D list with 0 (free space) and 1 (wall)
    start: tuple (x, y) - Death Eater's current position
    goal: tuple (x, y) - Harry's current position
    Returns: tuple (x, y) - Next position Death Eater should move to
    """
    queue = deque()
    queue.append(start)

    came_from = {}  # To reconstruct path
    came_from[start] = None

    while queue:
        current = queue.popleft()

        # If we reached Harry's position
        if current == goal:
            break

        for neighbor in get_neighbors(current, maze, cup_pos):
            if neighbor not in came_from:  # Not visited yet
                queue.append(neighbor)
                came_from[neighbor] = current

    # Reconstruct the path from goal to start
    if goal not in came_from:
        return start  # No path found, stay in place

    path = []
    current = goal
    while current != start:
        path.append(current)
        current = came_from[current]
    path.reverse()  # From start to goal

    # Return the first step in the path
    return path[0] if path else start



# Supposed to return the new coordinates of the character if a action is valid on a given state, or false
def is_action_valid(char_state, action):
    # Unpack current position
    x, y = char_state
    # Define movement deltas for each action
    # 0: Up, 1: Down, 2: Left, 3: Right
    dx_dy = [(-1, 0), (1, 0), (0, -1), (0, 1)]   
    dx, dy = dx_dy[action]
    new_x, new_y = x + dx, y + dy

    # Check if new position is within maze bounds
    if not (0 <= new_x < len(maze) and 0 <= new_y < len(maze[0])):
        return False
    # Check if the new position is not a wall
    if maze[new_x][new_y] == 1:  # assuming 1 means wall
        return False
    # If both checks pass, return the new position
    return (new_x, new_y)

def moving_average(data, window_size=50):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')


'''
------------------------------------------------------------------SETUP---------------------------------------------------------------
'''
# Loading the maze
maze = load_maze("maze.txt")
for i in range(10):
    for j in range(15):
        if maze[i][j] == 'X':
            maze[i][j] = 1
        else:
            maze[i][j] = 0
actions = [0, 1, 2, 3]  # 0=up, 1=down, 2=left, 3=right
# Q-table: maps (state) -> {action: Q-value}
q_table = defaultdict(lambda: {a: 0.0 for a in actions})
# only makes a entry when we access a particular state, hence it saves us from assigning a state to the walls, which are never going to be reached.

'''
---------------------------------------------------------------------TRAINING-------------------------------------------------------
'''

# Hyperparameters
alpha = 0.001         # Learning rate
gamma = 0.999         # Discount factor
epsilon = 1.0       # Exploration rate
min_epsilon = 0.001
decay_rate = 0.99999 # The rate which epsilon decays
num_episodes = 1000000
total_reward_log = [] # Keeps a track of total rewards in each episode
success_count = 0 # Required to count successive wins
generations_to_win = [] # Logs every episode where 10 consecutive wins are achieved
success_log = [] # Logs 1 if harry wins, 0 if it loses
step_count_per_episode = [] 

# TRAINING LOOP
for episode in range(1,num_episodes+1):
    state = reset(maze)
    harry_pos = (state[0],state[1])
    death_pos = (state[2], state[3])
    cup_pos = (state[4], state[5])
    total_reward = 0
    done = False
    number_of_steps_per_episode = 0
    
    while not done:
        new_harry_pos, action = harry_action(q_table, state, epsilon)
        new_death_pos = death_action(maze, death_pos, new_harry_pos, cup_pos)
        new_state = update_state(state, new_harry_pos, new_death_pos)
        number_of_steps_per_episode += 1

        if new_harry_pos == cup_pos:
            reward = +3000
            step_count_per_episode.append(number_of_steps_per_episode)
            done = True
            success_log.append(1)
            success_count += 1
        elif new_harry_pos == new_death_pos:
            reward = -1000
            step_count_per_episode.append(number_of_steps_per_episode)
            done = True
            success_log.append(0)
            success_count = 0
        else:

            reward = -1 -(episode//200000)  # per step penalty  
            

        if success_count ==10:
            generations_to_win.append(episode)
            success_count = 0
        q_table = q_update(q_table, state, action, reward,new_state, alpha, gamma)
        harry_pos = new_harry_pos
        death_pos = new_death_pos
        state = new_state
        total_reward += reward
    epsilon = max(min_epsilon, epsilon*decay_rate)
    total_reward_log.append(total_reward)


'''
-------------------------------------------------------------------EVALUATION-----------------------------------------------------
'''
# During training, all the logs were used to plot graphs.

window_size = 40000
# Plot moving average of success_log
plt.figure(figsize=(15, 4))
plt.subplot(1, 3, 1)
plt.plot(moving_average(success_log, window_size))
plt.title("Moving Average of Success Rate")
plt.xlabel("Episode")
plt.ylabel("Success Rate")

# Plot moving average of reward
plt.subplot(1, 3, 2)
plt.plot(moving_average(total_reward_log, window_size))
plt.title("Moving Average of Total Reward")
plt.xlabel("Episode")
plt.ylabel("Total Reward")

# Plot moving average of steps per episode
plt.subplot(1, 3, 3)
plt.plot(moving_average(step_count_per_episode, window_size))
plt.title("Moving Average of Steps per Episode")
plt.xlabel("Episode")
plt.ylabel("Steps")

'''
-------------------------------------------------------------------SAVING THE Q TABLE--------------------------------------------------------
'''
import pickle

q_table_regular = dict(q_table)

with open("trained_q_table.pkl", "wb") as f:
    pickle.dump(q_table_regular, f)





