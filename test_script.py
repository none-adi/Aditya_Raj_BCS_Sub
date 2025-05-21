import pygame
import pickle
from random import randint
import random
import numpy as np
from collections import deque, defaultdict

# -------------------------------------------------------------
# ------------------------ SUPPORT FUNCTIONS ------------------
# -------------------------------------------------------------

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

recent_states = deque(maxlen=10)  # track last 10 Harry positions

def harry_action(Q, state, epsilon):
    global recent_states
    harry_state_current = (state[0], state[1])

    # --- Exploration ---
    if random.random() < epsilon:
        actions = [0, 1, 2, 3]
        random.shuffle(actions)
        for action in actions:
            next_harry_state = is_action_valid(harry_state_current, action)
            if next_harry_state and next_harry_state not in recent_states:
                recent_states.append(next_harry_state)
                return next_harry_state, action

    # --- Exploitation ---
    q_actions = sorted(Q[state].items(), key=lambda x: x[1], reverse=True)
    for action, _ in q_actions:
        next_harry_state = is_action_valid(harry_state_current, action)
        if next_harry_state and next_harry_state not in recent_states:
            recent_states.append(next_harry_state)
            return next_harry_state, action

    # --- Fallback: allow revisiting if all options exhausted ---
    for action, _ in q_actions:
        next_harry_state = is_action_valid(harry_state_current, action)
        if next_harry_state:
            recent_states.append(next_harry_state)
            return next_harry_state, action

    # No valid move found, stay in place
    return harry_state_current, None

def update_state(state, harry_new_state, death_new_state):
    HX,HY,DX,DY,CX,CY = state
    hx,hy = harry_new_state
    dx, dy = death_new_state
    return (hx,hy,dx,dy,CX,CY)

def get_neighbors(pos, maze, cup_pos):
    neighbors = []
    x, y = pos
    rows = len(maze)
    cols = len(maze[0])

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < rows and 0 <= ny < cols and maze[nx][ny] != 1 and (nx, ny) != cup_pos:
            neighbors.append((nx, ny))

    return neighbors

def death_action(maze, start, goal, cup_pos):
    queue = deque()
    queue.append(start)

    came_from = {start: None}

    while queue:
        current = queue.popleft()
        if current == goal:
            break

        for neighbor in get_neighbors(current, maze, cup_pos):
            if neighbor not in came_from:
                queue.append(neighbor)
                came_from[neighbor] = current

    if goal not in came_from:
        return start  # No path found

    path = []
    current = goal
    while current != start:
        path.append(current)
        current = came_from[current]
    path.reverse()

    return path[0] if path else start

def is_action_valid(char_state, action):
    x, y = char_state
    dx_dy = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    dx, dy = dx_dy[action]
    new_x, new_y = x + dx, y + dy

    if not (0 <= new_x < len(maze) and 0 <= new_y < len(maze[0])):
        return False
    if maze[new_x][new_y] == 1:
        return False
    return (new_x, new_y)

def moving_average(data, window_size=50):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# ---------------------------- SETUP ----------------------------
maze = load_maze("maze.txt")
for i in range(10):
    for j in range(15):
        if maze[i][j] == 'X':
            maze[i][j] = 1
        else:
            maze[i][j] = 0

actions = [0, 1, 2, 3]

pygame.init()
WIDTH, HEIGHT = 500, 500
ROWS, COLS = len(maze), len(maze[0])
CELL_SIZE = WIDTH // COLS
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Goblet of Fire")
clock = pygame.time.Clock()

def draw_grid(maze, harry, death, cup):
    for i in range(len(maze)):
        for j in range(len(maze[0])):
            rect = pygame.Rect(j*CELL_SIZE, i*CELL_SIZE, CELL_SIZE, CELL_SIZE)
            color = (0, 0, 0) if maze[i][j] == 1 else (255, 255, 255)
            pygame.draw.rect(screen, color, rect)

            if (i, j) == harry:
                pygame.draw.circle(screen, (0, 0, 255), rect.center, CELL_SIZE//3)
            elif (i, j) == death:
                pygame.draw.circle(screen, (255, 0, 0), rect.center, CELL_SIZE//3)
            elif (i, j) == cup:
                pygame.draw.circle(screen, (0, 255, 0), rect.center, CELL_SIZE//3)

            pygame.draw.rect(screen, (200, 200, 200), rect, 1)


def run_game():
    recent_states.clear()

    with open("trained_q_table.pkl", "rb") as f:
        raw_q = pickle.load(f)
    Q = defaultdict(lambda: {0: 0, 1: 0, 2: 0, 3: 0}, raw_q)

    state = reset(maze)
    harry = (state[0], state[1])
    death = (state[2], state[3])
    cup = (state[4], state[5])

    done = False
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        new_harry, action = harry_action(Q, state, 0)
        death = death_action(maze, death, new_harry, cup)

        harry = new_harry
        state = update_state(state, harry, death)

        if harry == cup:
            print("Harry reached the cup!")
            done = True
        elif harry == death:
            print("Harry was caught!")
            done = True

        screen.fill((0, 0, 0))
        draw_grid(maze, harry, death, cup)
        pygame.display.flip()
        pygame.time.delay(400)
        clock.tick(60)

    pygame.time.delay(600)
    pygame.quit()

if __name__ == "__main__":
    run_game()
