####################################
#      YOU MAY EDIT THIS FILE      #
# MOST OF YOUR CODE SHOULD GO HERE #
####################################

# Imports from external libraries
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
import seaborn as sns

# Imports from this project
import constants
import config

# Configure matplotlib for interactive mode
plt.ion()

# Set the seaborn style
sns.set_theme(style="darkgrid", context="talk")


# The Robot class (which could be called "Agent") is the "brain" of the robot, and is used to decide what action to execute in the environment
class Robot:

    # Initialise a new robot
    def __init__(self, forward_kinematics):
        # Give the robot the forward kinematics function, to calculate the hand position from the state
        self.forward_kinematics = forward_kinematics
        # A list of visualisations which will be displayed on the right-side of the window
        self.planning_visualisation_lines = []
        self.model_visualisation_lines = []
        # The position of the robot's base
        self.robot_base_pos = np.array(constants.ROBOT_BASE_POS)
        # The goal state
        self.goal_state = 0
        # The number of steps in the episode
        self.num_steps = 0

    # Reset the robot at the start of an episode
    def reset(self):
        self.num_steps = 0

    # Give the robot access to the goal state
    def set_goal_state(self, goal_state):
        self.goal_state = goal_state

    # Function to get the next action in the plan
    def select_action(self, state):
        # For now, a random action, biased towards 'moving right'
        action = np.random.uniform(low=-constants.MAX_ACTION_MAGNITUDE, high=0.5*constants.MAX_ACTION_MAGNITUDE, size=2)
        self.num_steps += 1
        episode_done = False
        return action, episode_done

    # Function to add a transition to the buffer
    def add_transition(self, state, action, next_state):
        pass


# The VisualisationLine class enables us to store a line segment which will be drawn to the screen
class VisualisationLine:
    # Initialise a new visualisation (a new line)
    def __init__(self, x1, y1, x2, y2, colour, width):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.colour = colour
        self.width = width
