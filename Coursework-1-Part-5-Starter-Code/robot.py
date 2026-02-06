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

# Imports from this project
import constants
import config

# Configure matplotlib for interactive mode
plt.ion()


# The Robot class (which could be called "Agent") is the "brain" of the robot, and is used to decide what action to execute in the environment
class Robot:

    # Initialise a new robot
    def __init__(self, forward_kinematics):
        # Get the forward kinematics function from the environment
        self.forward_kinematics = forward_kinematics
        # A list of visualisations which will be displayed in the middle (planning) and right (policy) of the window
        self.demonstration_visualisation_lines = []
        self.policy_visualisation_lines = []
        # The replay buffer to store the transition data
        self.replay_buffer = ReplayBuffer()
        # The policy to be trained
        self.policy = Policy()
        # The number of steps in the episode/demonstration
        self.num_steps = 0
        # Flag to check if policy has been trained
        self.trained = False
        self.demos = []

    # Reset the robot at the start of an episode
    def reset(self):
        self.num_steps = 0
        self.prev_state = None # Reset previous state to avoid drawing lines between episodes
        # self.trained = False # Do not reset trained status!
        # self.demonstration_visualisation_lines = [] # Keep demo visualization!
        self.policy_visualisation_lines = []

    # Get the demonstrations
    def get_demos(self, demonstrator):
        # Only collect demos and train ONCE
        if not self.trained:
            demos = []
            for demo_num in range(config.NUM_DEMOS):
                print(f'Generating demonstration {demo_num+1} ...')
                # Generate demonstration
                demo = demonstrator.generate_demonstration()
                demos.append(demo)
                for state, action in demo:
                    self.replay_buffer.add_data(state, action)
            print(f'All demonstrations generated.')
            # Train the policy on all the demos
            print('Training ...')
            self.policy.train(self.replay_buffer)
            print('Training done.')
            self.trained = True
            self.demos = demos # Save demos for visualization
        
        # Visualise the demos (drawn once)
        if len(self.demonstration_visualisation_lines) == 0:
            self.create_demo_visualisations(self.demos)
        
        # Visualise the policy (vector field on right)
        # self.create_policy_visualisations()

    # Get the next action
    def select_action(self, state):
        episode_done = False
        action = self.policy.predict_next_action(state)
        
        # Check if the previous episode has finished
        if self.num_steps >= config.EPISODE_LENGTH:
            episode_done = True
            action = np.zeros(2)
        
        # Figure 5 Visualization: Draw policy path on Demonstration window
        if self.num_steps > 0:
            # Get previous and current hand positions
            # Safe check: ensure we have a valid prev_state (from THIS episode)
            if self.prev_state is not None:
                hand_pos_prev = self.forward_kinematics(self.prev_state)[2]
                hand_pos_curr = self.forward_kinematics(state)[2]
                
                # Draw RED line for policy execution
                # Append to demonstration_visualisation_lines so it appears on the middle window
                line = VisualisationLine(hand_pos_prev[0], hand_pos_prev[1], 
                                         hand_pos_curr[0], hand_pos_curr[1], 
                                         (255, 0, 0), 0.005) 
                self.demonstration_visualisation_lines.append(line)
        
        # Update prev_state and steps
        self.prev_state = state
        if not episode_done:
            self.num_steps += 1
            
        return action, episode_done

    # Function to create visualisations for the demonstrations
    def create_demo_visualisations(self, demos):
        # Loop over all demos
        for demo in demos:
            # Get the states for this demo
            states = [pair[0] for pair in demo]
            curr_state = states[0]
            # Loop over all steps in the demo
            for next_state in states[1:]:
                # Get the position of the start of the step
                hand_pos_prev = self.forward_kinematics(curr_state)[2]
                x1 = hand_pos_prev[0]
                y1 = hand_pos_prev[1]
                # Get the position of the end of the step
                hand_pos_next = self.forward_kinematics(next_state)[2]
                x2 = hand_pos_next[0]
                y2 = hand_pos_next[1]
                # Create a visualisation object for this step and add it to the list
                visualisation_line = VisualisationLine(x1, y1, x2, y2, (200, 200, 200), 0.005)
                self.demonstration_visualisation_lines.append(visualisation_line)
                # Update the current state
                curr_state = next_state

    # Function to create visualisations for the policy
    def create_policy_visualisations(self):
        # Gridlines
        colour = (150, 150, 150)
        width = 0.005
        # The grid is in joint space, which is 2D: theta1 and theta2
        # We can visualise the policy over this 2D space
        # But wait, the policy output is an action (change in joint angles)
        # Visualising a vector field in hand-space is hard because state -> hand is non-linear
        # Let's visualise the policy action as a line from hand_pos(state) to hand_pos(state + action)
        
        # Grid over state space (theta1, theta2)
        # Range of joint angles? Usually -pi to pi or similar
        # Let's assume a reasonable range or check constants if available. 
        # Actually, tutorial used 0 to 1 for x,y. Here it's angles.
        # Let's just create a grid from -3 to 3 for both angles (approx -pi to pi)
        for theta1 in np.linspace(-3, 3, 20):
            for theta2 in np.linspace(-3, 3, 20):
                state = np.array([theta1, theta2])
                
                # Predict action
                action = self.policy.predict_next_action(state)
                next_state = state + action
                
                # Forward kinematics to get hand positions
                hand_pos_curr = self.forward_kinematics(state)[2]
                hand_pos_next = self.forward_kinematics(next_state)[2]
                
                # Draw line
                line = VisualisationLine(hand_pos_curr[0], hand_pos_curr[1], 
                                         hand_pos_next[0], hand_pos_next[1], 
                                         (0, 200, 0), 0.005) # Green for policy
                self.policy_visualisation_lines.append(line)


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


# Policy is used to predict the next action
class Policy:

    def __init__(self):
        self.network = Network()
        self.optimiser = optim.Adam(self.network.parameters(), lr=config.LEARNING_RATE)
        self.loss_fn = nn.MSELoss()
        self.losses = []
        self.fig, self.ax = plt.subplots(num="Loss Curve", clear=True)
        self.ax.set_xlabel('Num Epochs')
        self.ax.set_ylabel('Loss')
        self.ax.set_title('Loss Curve')
        self.ax.set_yscale('log')
        self.line, = self.ax.plot([], [], linestyle='-', marker=None, color='blue')
        plt.show()

    def train(self, buffer):
        # Train for a certain number of epochs
        for epoch_num in range(config.NUM_TRAINING_EPOCHS):
            #
            loss_sum = 0
            epoch_minibatches = buffer.sample_epoch_minibatches(config.MINIBATCH_SIZE)
            for inputs, targets in epoch_minibatches:
                # Set the network to training mode
                self.network.train()
                # Forward pass
                predictions = self.network.forward(inputs)
                # Compute the loss
                loss = self.loss_fn(predictions, targets)
                # Backward pass and optimization step
                self.optimiser.zero_grad()
                loss.backward()
                self.optimiser.step()
                # Accumulate the loss
                loss_sum += loss.item()
            ave_loss = loss_sum / len(epoch_minibatches)
            self.losses.append(ave_loss)
            # Plot the loss curve
            self.line.set_xdata(range(1, len(self.losses) + 1))
            self.line.set_ydata(self.losses)
            # Adjust the plot limits
            self.ax.relim()
            self.ax.autoscale_view()
            # Redraw the figure
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

    def predict_next_action(self, state):
        # Convert to torch tensor
        input = torch.from_numpy(state).float()
        # Set model to evaluation mode
        self.network.eval()
        with torch.no_grad():
            # Forward pass
            prediction_tensor = self.network(input)
        # Remove batch dimension and convert to numpy
        prediction = prediction_tensor.squeeze(0).numpy()
        return prediction


# This is the network that is trained on the transition data
class Network(nn.Module):

    # Initialise
    def __init__(self, input_size=2, hidden_size=20, output_size=2):
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
        # Pass data through output layer
        output = self.output(x)
        return output


# ReplayBuffer class stores transitions
class ReplayBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.size = 0

    def add_data(self, state, action):
        self.states.append(state)
        self.actions.append(action)
        self.size += 1

    # Create minibatches for a single epoch of training (one epoch means all the training data is seen once)
    def sample_epoch_minibatches(self, minibatch_size):
        # Convert lists to NumPy arrays for indexing
        states_array = np.array(self.states)
        actions_array = np.array(self.actions)
        # Shuffle indices
        indices = np.random.permutation(self.size)
        minibatches = []
        # Create minibatches
        for i in range(0, self.size, minibatch_size):
            # Get the indices for this minibatch
            minibatch_indices = indices[i: i + minibatch_size]
            minibatch_states = states_array[minibatch_indices]
            minibatch_actions = actions_array[minibatch_indices]
            # Convert to torch tensors
            inputs = torch.tensor(minibatch_states, dtype=torch.float32)
            targets = torch.tensor(minibatch_actions, dtype=torch.float32)
            minibatches.append((inputs, targets))
        return minibatches
