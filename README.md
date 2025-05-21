# Goblet of Fire 

##  Basic Approach

1. **Maze Structure**: The maze is loaded from a text file (`maze.txt`) where:
   - `'X'` represents walls, and blanks represent movable blocks
   - The maze is read from the file and converted into a 2D grid of `0`s and `1`s.

2. **Q-Learning Agent (Harry)**:
   - The State and the Actions for Harry:
     - State: `(harry_x, harry_y, death_x, death_y, cup_x, cup_y)`
        - Where charac_x, charac_y represent the x and y coordinates for the given character.

        - While creating the state, we ensure that they have unique random positions.

     - Actions: `actions = [0, 1, 2, 3]`
          - where `0` means **UP**, `1` means **DOWN**, `2` means **LEFT** ,`3` means **RIGHT**.

        - At every point of time, not all actions are legal, hence only those actions which are legal will be allowed.
        
   - Chooses an action based on the ε-greedy strategy.
        - The Agent chooses a random action with epsilon probability.
        - With 1-epsilon probability, it choses the best action from the q_table

   - Q-values are updated using the standard Q-learning formula, in a nested dictionary which has the following structure:
        ```bash
        Q[state][action] = Q_value
        ```
        - This dictionary is made using the `defaultdict` class from the `collections` module, which allows us to add default values if we ever send keys which are not in the dictionary

        - Hence, whenever the model encounters a state, it is appended into the Q table with default value for all actions as `0`.

3. **Death Eater**:
   - Uses **Breadth-First Search (BFS)** to always find the shortest path to Harry.
   - Can never reach the location of the Cup(illegal move).

4. **Reward Structure**:
   - Harry gets a positive reward of 3000 for reaching the cup.
   - A negative reward of 1000 for being caught.
   - Small step penalty(which is given by -1-(episode//200000)) encourages quicker escapes. We increase it as number of episodes increase as initially that the model needs to explore and shouldn't be penalized too much for steps, but with time it should start chosing smaller routes.


5. **Training**:
   - The Agent is trained on One Million episodes.
   - The trained Q-table is saved using `pickle`.

6. **Testing**
    - No training happens here, Harry choses his action based on the highest q_value associated with a state and action
    - Harry isn't allowed to make moves which are repetitive. This ensures that our model does not gets stuck in an infinite loop. This is enforced by keeping a track of last 10 recently visited states, and ensuring the model does not keeps repeating the same moves.
    - The maze is loaded from the given file.
            

6. **Visualization**:
   - A Pygame window visualizes Harry (blue), the Death Eater (red), and the Cup (green).
   - The agent plays according to the learned policy from the Q-table.
   - The game ends when either Harry dies or gets the cup

## Installation and Setup
- Folder Structure
    - `maze.txt`
    - `test_script.py`
    - `trained_q_table.pkl`
    - `training.py`
    - `README.md`
    - `requirements.txt`

- Make sure all these files are installed along with the latest version of python.
- To install all the dependencies, run:
   ```bash
   pip install -r requirements.txt
   ```

- Finally, to run the pygame, execute the file `test_script.py`.




