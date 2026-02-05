##########################
# YOU MAY EDIT THIS FILE #
##########################

# The random seed for numpy.
# Setting this to 0 means that it will be different each time.
RANDOM_SEED = 0

# The window width and height in pixels, for both the "environment" window and the "planning" window.
# If you wish, you can modify this according to the size of your screen.
WINDOW_SIZE = 400

# The frame rate for pygame, which determines how quickly the program runs.
# Specifically, this is the number of time steps per second that the robot will execute an action in the environment.
# You may wish to slow this down to observe the robot's movement, or speed it up to run large-scale experiments.
FRAME_RATE = 10

# You may want to add your own configuration variables here, depending on the algorithm you implement.
# CEM Parameters
CEM_NUM_ITER = 10
CEM_NUM_PATHS = 50
CEM_EPISODE_LENGTH = 50
CEM_NUM_ELITES = 5
