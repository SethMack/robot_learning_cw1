####################################
#      YOU MAY EDIT THIS FILE      #
# MOST OF YOUR CODE SHOULD GO HERE #
####################################

# Imports from external libraries
import numpy as np

# Imports from this project
import constants
import config


# The Robot class (which could be called "Agent") is the "brain" of the robot, and is used to decide what action to execute in the environment
class Robot:

    # Initialise a new robot
    def __init__(self):
        # The environment
        self.environment = None
        # The number of steps in the episode so far
        self.num_steps = 0
        # A list of visualisations which will be displayed on the right-side of the window
        self.visualisation_lines = []
        self.visualisation_circles = []

    # Reset the robot at the start of an episode
    def reset(self):
        self.num_steps = 0
        self.visualisation_lines = []
        self.visualisation_circles = []

    # Give the robot access to the environment
    def set_environment(self, environment):
        self.environment = environment

    # Function to get the next action in the plan
    def select_action(self, state):
        # If we haven't planned yet, do the planning
        if self.num_steps == 0:
            self.cem_planning(state)
        
        # Get the next action in the plan
        if self.num_steps < len(self.planned_actions):
            action = self.planned_actions[self.num_steps]
        else:
            # Should not happen if logic is correct, but safe fallback
            action = np.zeros(2)

        # Increment the number of steps executed so far
        self.num_steps += 1
        
        # Check if the episode has finished
        if self.num_steps >= len(self.planned_actions):
            episode_done = True
        else:
            episode_done = False
            
        return action, episode_done

    # Planning with cross-entropy method
    def cem_planning(self, state):
        # Create some placeholders for the data
        # Actions: [Iter, Path, Step, Dim]
        sampled_actions = np.zeros([config.CEM_NUM_ITER, config.CEM_NUM_PATHS, config.CEM_EPISODE_LENGTH, 2], dtype=np.float32)
        # Store mean and std for each iteration
        action_mean = np.zeros([config.CEM_NUM_ITER, config.CEM_EPISODE_LENGTH, 2], dtype=np.float32)
        action_std = np.zeros([config.CEM_NUM_ITER, config.CEM_EPISODE_LENGTH, 2], dtype=np.float32)

        # Loop over Iterations
        for iter_num in range(config.CEM_NUM_ITER):
            distances = np.zeros(config.CEM_NUM_PATHS, dtype=np.float32)
            
            # Loop over Paths
            for path_num in range(config.CEM_NUM_PATHS):
                curr_state = state.copy()
                
                # Loop over Steps
                for step in range(config.CEM_EPISODE_LENGTH):
                    # Sample Action
                    if iter_num == 0:
                        # First iteration: Uniform distribution
                        action = np.random.uniform(low=-constants.MAX_ACTION_MAGNITUDE, high=constants.MAX_ACTION_MAGNITUDE, size=2)
                    else:
                        # Subsequent iterations: Normal distribution from previous mean/std
                        action = np.random.normal(loc=action_mean[iter_num-1, step], scale=action_std[iter_num-1, step])
                        # Clip action to be within bounds
                        action = np.clip(action, -constants.MAX_ACTION_MAGNITUDE, constants.MAX_ACTION_MAGNITUDE)
                    
                    # Store action
                    sampled_actions[iter_num, path_num, step] = action
                    
                    # Execute dynamics
                    next_state = self.environment.dynamics(curr_state, action)
                    curr_state = next_state
                
                # Calculate Distance (Negative Euclidean between Final Hand Pos and Goal)
                # The goal_state is the hand position (red circle)
                final_hand_pos = self.environment.get_joint_pos_from_state(curr_state)[2]
                distances[path_num] = np.linalg.norm(final_hand_pos - self.environment.goal_state)
            
            # Select Elites (lowest distances)
            elites_indices = np.argsort(distances)[:config.CEM_NUM_ELITES]
            elite_actions = sampled_actions[iter_num, elites_indices]
            
            # Update Mean and Std for this iteration
            action_mean[iter_num] = np.mean(elite_actions, axis=0)
            action_std[iter_num] = np.std(elite_actions, axis=0)

        # Set the planned actions to be the mean of the final iteration
        self.planned_actions = action_mean[-1]
        
        # VISUALISATION: Draw the mean path for each iteration
        self.visualisation_lines = []
        for iter_num in range(config.CEM_NUM_ITER):
            curr_state = state.copy()
            # Get initial hand position
            hand_pos_prev = self.environment.get_joint_pos_from_state(curr_state)[2]
            
            # Determine colour based on iteration (Dark -> Bright)
            intensity = (iter_num + 1) / config.CEM_NUM_ITER
            brightness = int(50 + 205 * intensity)
            colour = (brightness, brightness, brightness)
            
            # Initial width
            width = 0.002 + 0.003 * intensity

            for step in range(config.CEM_EPISODE_LENGTH):
                action = action_mean[iter_num, step]
                next_state = self.environment.dynamics(curr_state, action)
                hand_pos_curr = self.environment.get_joint_pos_from_state(next_state)[2]
                
                # Create visualisation line for the hand trace
                self.visualisation_lines.append(VisualisationLine(
                    hand_pos_prev[0], hand_pos_prev[1], 
                    hand_pos_curr[0], hand_pos_curr[1], 
                    colour=colour, width=width
                ))
                
                curr_state = next_state
                hand_pos_prev = hand_pos_curr


# The VisualisationLine class enables us to store a line segment which will be drawn to the screen
class VisualisationLine:
    # Initialise a new visualisation (a new line)
    def __init__(self, x1, y1, x2, y2, colour=(255, 255, 255), width=0.01):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.colour = colour
        self.width = width


# The VisualisationCircle class enables us to a circle which will be drawn to the screen
class VisualisationCircle:
    # Initialise a new visualisation (a new circle)
    def __init__(self, x, y, radius, colour=(255, 255, 255)):
        self.x = x
        self.y = y
        self.radius = radius
        self.colour = colour