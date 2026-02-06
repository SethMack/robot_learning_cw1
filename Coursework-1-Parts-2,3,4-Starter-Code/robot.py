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
        self.dynamics_model = DynamicsModel()
        self.replay_buffer = ReplayBuffer()
        self.num_episodes = 0

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
        # Determine whether to end the episode
        if self.num_steps == config.EPISODE_LENGTH:
            self.reset()
            self.num_episodes += 1
            episode_done = True
            if self.num_episodes % 20 == 0:
                print("Training dynamics model on episode {}".format(self.num_episodes))
                # Train the model on the buffer for a for epochs
                self.dynamics_model.train(self.replay_buffer, config.TRAIN_NUM_MINIBATCH)
                # Visualise the trained model
                # self.create_model_visualisations() #TODO: implement this function
        else:
            episode_done = False
        # Return the action, and a flag indicating if the episode has finished, to the main program loop
        return action, episode_done


    # Function to add a transition to the buffer
    def add_transition(self, state, action, next_state):
        self.replay_buffer.add_transition(state, action, next_state)


# This is the network that is trained on the transition data
class Network(nn.Module):

    # Initialise
    def __init__(self, input_size=4, hidden_size=64, output_size=2):
        super(Network, self).__init__()
        # Define the first hidden layer
        self.hidden1 = nn.Linear(input_size, hidden_size)
        self.activation1 = nn.ReLU()
        # Define the second hidden layer
        self.hidden2 = nn.Linear(hidden_size, hidden_size)
        self.activation2 = nn.ReLU()
        # Define the third hidden layer
        self.hidden3 = nn.Linear(hidden_size, hidden_size)
        self.activation3 = nn.ReLU()
        # Define an additional hidden layer
        self.hidden4 = nn.Linear(hidden_size, hidden_size)
        self.activation4 = nn.ReLU()
        # Define the output layer
        self.output = nn.Linear(hidden_size, output_size)

    # Forward pass
    def forward(self, input):
        # Pass data through the layers
        x = self.hidden1(input)
        x = self.activation1(x)
        x = self.hidden2(x)
        x = self.activation2(x)
        x = self.hidden3(x)
        x = self.activation3(x)
        x = self.hidden4(x)  # Additional layer
        x = self.activation4(x)
        # Pass data through output layer
        output = self.output(x)
        return output


# DynamicsModel is used to learn the environment dynamics
class DynamicsModel:

    def __init__(self):
        self.network = Network()
        self.optimiser = optim.Adam(self.network.parameters(), lr=0.001)  # Reduced learning rate for better convergence
        self.loss_fn = nn.MSELoss()
        self.losses = []
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlabel('Num Minibatches')
        self.ax.set_ylabel('Loss')
        self.ax.set_title('Loss Curve')
        self.ax.set_yscale('log')
        self.line, = self.ax.plot([], [], linestyle='-', marker=None, color='blue')
        plt.show()

    def train(self, buffer, num_minibatch):
        if len(buffer.data) < config.TRAIN_MINIBATCH_SIZE:
            return
        loss_sum = 0
        prev_loss = float('inf')
        for minibatch_num in range(num_minibatch):
            inputs, targets = buffer.sample_minibatch(config.TRAIN_MINIBATCH_SIZE)
            # Set the network to training mode
            self.network.train()
            # Forward pass: compute predicted next states
            predictions = self.network.forward(inputs)
            # Compute the loss
            loss = self.loss_fn(predictions, targets)
            # Backward pass and optimization step
            self.optimiser.zero_grad()
            loss.backward()
            self.optimiser.step()
            # Get the loss value and add to the list of losses
            loss_value = loss.item()
            loss_sum += loss_value
            # Check for convergence
            if abs(prev_loss - loss_value) < 1e-5:  # Convergence threshold
                print(f"Converged at minibatch {minibatch_num + 1} with loss {loss_value}")
                break
            prev_loss = loss_value
            
            # Interactive plot update
            if minibatch_num % 100 == 0:
                 self.update_plot()

        ave_loss = loss_sum / (minibatch_num + 1)
        self.losses.append(ave_loss)
        
        self.update_plot() # Final update

    def update_plot(self):
        self.line.set_xdata(range(len(self.losses)))
        self.line.set_ydata(self.losses)
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

    def predict_next_state(self, state, action):
        # Concatenate state and action
        input = np.concatenate((state, action))
        # Convert to torch tensors
        input = torch.from_numpy(input).float()
        # Set model to evaluation mode
        self.network.eval()
        with torch.no_grad():
            # Forward pass
            prediction_tensor = self.network(input)
        # Remove batch dimension and convert to numpy
        prediction = prediction_tensor.squeeze(0).numpy()
        return prediction

    def create_model_visualisations(self):
        # Clear previous visualisation lines
        self.model_visualisation_lines.clear()

        # Gridlines for the rightmost panel
        colour = (150, 150, 150)
        width = 0.005
        for row in range(10):
            y1 = y2 = row * 0.1
            x1 = 0.0
            x2 = 1.0
            line = VisualisationLine(x1, y1, x2, y2, colour, width)
            self.model_visualisation_lines.append(line)
        for col in range(10):
            x1 = x2 = col * 0.1
            y1 = 0.0
            y2 = 1.0
            line = VisualisationLine(x1, y1, x2, y2, colour, width)
            self.model_visualisation_lines.append(line)

        # Draw each cell with actions and predicted next states
        for row in range(10):
            for col in range(10):
                # Calculate the centre of the cell, i.e. the state
                x = col * 0.1 + 0.05
                y = row * 0.1 + 0.05
                state = np.array([x, y])
                # Draw a red line for action [0.0, 0.04]
                action = np.array([0.0, 0.04])
                next_state = self.dynamics_model.predict_next_state(state, action)
                line = VisualisationLine(x, y, next_state[0], next_state[1], (255, 0, 0), 0.005)
                self.model_visualisation_lines.append(line)
                # Draw a green line for action [0.04, 0.0]
                action = np.array([0.04, 0.0])
                next_state = self.dynamics_model.predict_next_state(state, action)
                line = VisualisationLine(x, y, next_state[0], next_state[1], (0, 200, 0), 0.005)
                self.model_visualisation_lines.append(line)
                # Draw a blue line for action [0.0, -0.04]
                action = np.array([0.0, -0.04])
                next_state = self.dynamics_model.predict_next_state(state, action)
                line = VisualisationLine(x, y, next_state[0], next_state[1], (0, 0, 255), 0.005)
                self.model_visualisation_lines.append(line)
                # Draw a yellow line for action [-0.04, 0.0]
                action = np.array([-0.04, 0.0])
                next_state = self.dynamics_model.predict_next_state(state, action)
                line = VisualisationLine(x, y, next_state[0], next_state[1], (200, 200, 0), 0.005)
                self.model_visualisation_lines.append(line)

        # Plot the loss curve in a separate figure
        self.plot_loss_curve()

    def plot_loss_curve(self):
        plt.figure()
        plt.plot(range(1, len(self.losses) + 1), self.losses, linestyle='-', marker=None, color='blue')
        plt.xlabel('Num Minibatches')
        plt.ylabel('Loss')
        plt.title('Loss Curve')
        plt.yscale('log')
        plt.show()


# ReplayBuffer class stores transitions
class ReplayBuffer:
    def __init__(self):
        self.data = []
        self.size = 0

    def add_transition(self, state, action, next_state):
        self.data.append((state, action, next_state))
        self.size += 1

    def sample_minibatch(self, minibatch_size):
        if minibatch_size > self.size:
            raise ValueError("Minibatch size larger than the buffer size.")
        # Randomly sample indices for this minibatch
        indices = np.random.choice(self.size, minibatch_size, replace=False)
        minibatch = [self.data[i] for i in indices]
        # Unzip the minibatch
        states, actions, next_states = zip(*minibatch)
        # Concatenate states and actions along the feature dimension
        inputs = np.concatenate([states, actions], axis=1)
        targets = np.array(next_states)
        # Convert to torch tensors
        inputs = torch.from_numpy(inputs).float()
        targets = torch.from_numpy(targets).float()
        return inputs, targets





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
