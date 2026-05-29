import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random
import pickle
import time




# ---------------------------------------------------------
# Linear layer
# ---------------------------------------------------------
class Linear:
    """A fully connected layer implemented with NumPy arrays."""

    def __init__(
        self, in_features: int, out_features: int, batch_size: int
    ) -> None:
        super(Linear, self).__init__()
        self.batch_size = batch_size
        self.weight = np.random.normal(size=(in_features, out_features)) * np.sqrt(
            1.0 / in_features
        )
        self.bias = np.random.normal(size=(out_features,)) * np.sqrt(1.0 / in_features)
        self.grad_weight = np.zeros((in_features, out_features))
        self.grad_bias = np.zeros(out_features)
        self.input = np.zeros((batch_size, in_features))

    def forward(self, input: np.ndarray) -> np.ndarray:
        self.input = input
        output = input @ self.weight + self.bias
        return output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        
        # Reshape in case the input is flattened
        grad_output = grad_output.reshape(-1, np.transpose(self.weight).shape[0])  
        
        grad_input = grad_output @ np.transpose(self.weight)
        self.grad_weight = np.transpose(self.input) @ grad_output
        self.grad_bias = np.sum(grad_output, axis=0)
        return grad_input

    def update(self, lr) -> None:
        self.weight = self.weight - lr * self.grad_weight / self.batch_size
        self.bias = self.bias - lr * self.grad_bias / self.batch_size

    def get_weights(self):
        # returns the weights and bias
        return [self.weight, self.bias]

    def set_weights(self, new_weights):
        # sets weights and bias of the neural network
        self.weight = new_weights[0]
        self.bias = new_weights[1]


# ---------------------------------------------------------
# Sigmoid activation
# ---------------------------------------------------------
class Sigmoid:
    """Sigmoid activation function"""

    def __init__(self, in_features: int, batch_size: int) -> None:
        super(Sigmoid, self).__init__()
        self.input = np.zeros(batch_size)
        self.output = np.zeros(batch_size)

    def forward(self, input: np.ndarray) -> np.ndarray:
        self.input = input
        output = 1 / (1 + np.exp(-self.input))
        self.output = output
        return output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        grad_input = grad_output * (self.output * (1 - self.output))
        return grad_input


# ---------------------------------------------------------
# ReLU activation
# ---------------------------------------------------------
class Relu:
    """ReLU activation function"""

    def __init__(self, in_features: int, batch_size: int) -> None:
        super(Relu, self).__init__()
        self.input = np.zeros(batch_size)
        self.output = np.zeros(batch_size)

    def forward(self, input: np.ndarray) -> np.ndarray:
        self.input = input
        output = np.maximum(input,0)
        self.output = output
        return output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        grad_input = np.where(grad_output > 0, 1, 0)
        return grad_input

# Define model
class NN:
    def __init__(self, in_states, h1_nodes, out_actions, batch_size: int):
        super(NN, self).__init__()
        #super().__init__()

        # Define network layers
        self.in_features = in_states
        self.l1 = Linear(in_states, h1_nodes, batch_size)   # Linear layer
        self.s1 = Sigmoid(h1_nodes, batch_size) # Sigmoid layer
        self.l2 = Linear(h1_nodes, h1_nodes, batch_size)   # Linear layer
        self.s2 = Sigmoid(h1_nodes, batch_size) # Sigmoid layer
        self.l3 = Linear(h1_nodes, out_actions, batch_size) # Ouptut layer 


    def forward(self, x: np.ndarray) -> np.ndarray:
        
        x = np.array(x)
        x = x.reshape(x.shape[0], -1)  # Flatten the input
        x = x.T # TODO THAT DOESNT SEEM RIGHT BUT IT SEEMS TO WORK
        x = self.l1.forward(x)    # Linear layer
        x = self.s1.forward(x)    # Apply sigmoid activation function
        x = self.l2.forward(x)    # Linear layer
        x = self.s2.forward(x)    # Apply sigmoid activation function
        x = self.l3.forward(x)    # Linear layer
        x = softmax(x)         # Apply softmax

        return x[0]
    
    def backward(self, x: np.ndarray) -> None:
        x = self.l3.backward(x)
        x = self.s2.backward(x)
        x = self.l2.backward(x)
        x = self.s1.backward(x)
        x = self.l1.backward(x)
    
    def update(self,lr) -> None:
        self.l1.update(lr)
        self.l2.update(lr)
        self.l3.update(lr)

    def get_weights(self):
        # Returns weights of the neurons
        return [self.l1.get_weights(), self.l2.get_weights(), self.l3.get_weights()]

    def set_weights(self, w):
        # Sets weights of the neurons
        # TODO testing
        self.l1.set_weights(w[0])
        self.l2.set_weights(w[1])
        self.l3.set_weights(w[2])

    def print_weights(self):
        # Prints weights of the neurons
        # TODO testing
        print("First_layer: ", self.l1.get_weights()) 
        print("Second_layer: ", self.l2.get_weights()) 
        print("Third_layer: ", self.l3.get_weights())


# Define model
class NN_Relu:
    def __init__(self, in_states, h1_nodes, out_actions, batch_size: int):
        super(NN_Relu, self).__init__()
        #super().__init__()

        # Define network layers
        self.in_features = in_states
        self.l1 = Linear(in_states, h1_nodes, batch_size)   # Linear layer
        self.r1 = Relu(h1_nodes, batch_size) # Relu layer
        self.l2 = Linear(h1_nodes, h1_nodes, batch_size)   # Linear layer
        self.r2 = Relu(h1_nodes, batch_size) # Relu layer
        self.l3 = Linear(h1_nodes, out_actions, batch_size) # Ouptut layer 


    def forward(self, x: np.ndarray) -> np.ndarray:
        
        x = np.array(x)
        x = x.reshape(x.shape[0], -1)  # Flatten the input
        x = x.T # TODO THAT DOESNT SEEM RIGHT BUT IT SEEMS TO WORK
        x = self.l1.forward(x)    # Linear layer
        x = self.r1.forward(x)    # Apply relu activation function
        x = self.l2.forward(x)    # Linear layer
        x = self.r2.forward(x)    # Apply relu activation function
        x = self.l3.forward(x)    # Linear layer
        x = softmax(x)         # Apply softmax

        return x[0]
    
    def backward(self, x: np.ndarray) -> None:
        x = self.l3.backward(x)
        x = self.r2.backward(x)
        x = self.l2.backward(x)
        x = self.r1.backward(x)
        x = self.l1.backward(x)
    
    def update(self,lr) -> None:
        self.l1.update(lr)
        self.l2.update(lr)
        self.l3.update(lr)

    def get_weights(self):
        # Returns weights of the neurons
        return [self.l1.get_weights(), self.l2.get_weights(), self.l3.get_weights()]

    def set_weights(self, w):
        # Sets weights of the neurons
        # TODO testing
        self.l1.set_weights(w[0])
        self.l2.set_weights(w[1])
        self.l3.set_weights(w[2])

    def print_weights(self):
        # Prints weights of the neurons
        # TODO testing
        print("First_layer: ", self.l1.get_weights()) 
        print("Second_layer: ", self.l2.get_weights()) 
        print("Third_layer: ", self.l3.get_weights())


# ---------------------------------------------------------
# Utilities for training
# ---------------------------------------------------------
def softmax(input: np.ndarray) -> np.ndarray:
    """Compute the row-wise softmax of the input logits."""
    output = np.exp(input) / np.sum(np.exp(input), axis=1, keepdims=True)
    return output


def compute_loss(target: np.ndarray, prediction: np.ndarray) -> float:
    """Return the average cross-entropy loss for a batch of predictions."""
    return -np.sum(target * np.log(prediction+1e-8)) / prediction.shape[0] # TODO 1e-8

# def compute_loss(target: np.ndarray, prediction: np.ndarray) -> float:
#     """Return MSE"""
#     return np.sum((target -prediction)**2) / prediction.shape[0] 


def compute_gradient(target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    """
    Computes the gradient of the cross-entropy loss w.r.t. the predictions.
    The below formula is valid for softmax + cross-entropy loss with one-hot targets.
    Due to this, we do not need to implement a backward pass for the softmax layer.
    """
    return prediction - target

# def compute_gradient(target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
#     """
#     Computes the gradient of the cross-entropy loss w.r.t. the predictions.
#     The below formula is valid for softmax + cross-entropy loss with one-hot targets.
#     Due to this, we do not need to implement a backward pass for the softmax layer.
#     """
#     # TODO? ADD SOFTMAX GRADIENT
#     return 2*(target -prediction) / prediction.shape[0]


def one_hot(a: np.ndarray, num_classes: int) -> np.ndarray:
    return np.squeeze(np.eye(num_classes)[a.reshape(-1)])

# ---------------------------------------------------------    


# Define memory for Experience Replay
class ReplayMemory():
    def __init__(self, maxlen):
        self.memory = deque([], maxlen=maxlen)
    
    def append(self, transition):
        self.memory.append(transition)

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)

    def __len__(self):
        return len(self.memory)

# FrozeLake Deep Q-Learning
class FrozenLakeDQL():
    # Hyperparameters (adjustable)
    learning_rate_a = 0.1       # learning rate (alpha), default: 0.001 (tutorial) or 0.1 (empirical)
    discount_factor_g = 0.9         # discount rate (gamma), default: 0.9  
    network_sync_rate = 10          # number of steps the agent takes before syncing the policy and target network, default: 10
    replay_memory_size = 1_000       # size of replay memory, default: 1_000
    mini_batch_size = 32          # size of the training data set sampled from the replay memory, default: 32

    # Neural Network stuff
    def loss_fn(self,y_true, y_pred):
        return np.square(y_true-y_pred).mean()   # Loss function. MSE=Mean Squared Error can be swapped to something else.

    optimizer = None                # Optimizer. TODO 

    ACTIONS = ['L','D','R','U']     # for printing 0,1,2,3 => L(eft),D(own),R(ight),U(p)

    # Train the FrozeLake environment
    def train(self, episodes, render, is_slippery, relu = False, hidden_layer_size = 16):
        # Create FrozenLake instance
        #env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=is_slippery, render_mode='human' if render else None, reward_schedule=(1, 0, -0.01))
        
        curr_render_mode = 'human' if render else None
        env = gym.make(
            'FrozenLake-v1', 
            desc=["SFFF", "FFFF", "FFFF", "FFFG"], 
            # map_name="4x4",
            is_slippery=is_slippery, 
            render_mode=curr_render_mode,
            reward_schedule=(1, 0.0, -0.01) 
        )
        loss_list = []   

        num_states = env.observation_space.n
        num_actions = env.action_space.n
        
        epsilon = 1 # 1 = 100% random actions
        memory = ReplayMemory(self.replay_memory_size)

        # Create policy and target network. Number of nodes in the hidden layer can be adjusted.
        if(relu == True): 
            policy_dqn = NN_Relu(in_states=num_states, h1_nodes=hidden_layer_size, out_actions=num_actions, batch_size = self.mini_batch_size)

            target_dqn = NN_Relu(in_states=num_states, h1_nodes=hidden_layer_size, out_actions=num_actions, batch_size = self.mini_batch_size)
        else: 
            policy_dqn = NN(in_states=num_states, h1_nodes=hidden_layer_size, out_actions=num_actions, batch_size = self.mini_batch_size)

            target_dqn = NN(in_states=num_states, h1_nodes=hidden_layer_size, out_actions=num_actions, batch_size = self.mini_batch_size)

        # Make the target and policy networks the same (copy weights/biases from one network to the other)
        target_dqn.set_weights(policy_dqn.get_weights())

        # Printing pre training policy network
        # print('Policy (random, before training):')
        # policy_dqn.print_weights()

        # Policy network optimizer. "Adam" optimizer can be swapped to something else.
        # TODO HOW TO IMPLEMENT OPTIMIZER 
        # self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)

        # List to keep track of rewards collected per episode. Initialize list to 0's.
        rewards_per_episode = np.zeros(episodes)

        # List to keep track of epsilon decay
        epsilon_history = []

        # List to keep track of learning rate
        learning_rate_history = []

        # Track number of steps taken. Used for syncing policy => target network.
        step_count=0
            
        for i in range(episodes):
            if(i%500 == 0):
                print("Epoch: ", i)
            
            if(i >= np.floor(episodes/5) and i%(episodes/20) == 0):    # TODO arbitrary number, default: 3_000
                self.learning_rate_a = self.learning_rate_a/1.2 # default: 2

            learning_rate_history.append(self.learning_rate_a)

            state = env.reset()[0]  # Initialize to state 0
            terminated = False      # True when agent falls in hole or reached goal
            truncated = False       # True when agent takes more than 200 actions    

            # Agent navigates map until it falls into hole/reaches goal (terminated), or has taken 200 actions (truncated).
            while(not terminated and not truncated):

                # Select action based on epsilon-greedy
                if random.random() < epsilon:
                    # select random action
                    action = env.action_space.sample() # actions: 0=left,1=down,2=right,3=up
                else:
                    # select best action   
                    action = policy_dqn.forward(self.state_to_dqn_input(state, num_states)).argmax().item()

                # Execute action
                new_state,reward,terminated,truncated,_ = env.step(action)

                # Save experience into memory
                memory.append((state, action, new_state, reward, terminated)) 

                # Move to the next state
                state = new_state

                # Increment step counter
                step_count+=1

                rewards_per_episode[i] += reward

            # Keep track of the rewards collected per episode.
            # TODO CHANGED
            # if reward != 0:
            #rewards_per_episode[i] = reward

            # Check if enough experience has been collected and if at least 1 reward has been collected
            if len(memory)>self.mini_batch_size  and np.max(rewards_per_episode) > 0: # TODO and np.sum(rewards_per_episode)!=0
                mini_batch = memory.sample(self.mini_batch_size)
                loss_list.append(self.optimize(mini_batch, policy_dqn, target_dqn, self.learning_rate_a))        

                # Decay epsilon
                epsilon = max(epsilon - 1/episodes, 0.0) # TODO maybe do 0.1
                # epsilon = max(epsilon - 1/episodes/2, 0) # maybe if we want bigger epsilon later

                epsilon_history.append(epsilon)

                # Copy policy network to target network after a certain number of steps
                if step_count > self.network_sync_rate:
                    target_dqn.set_weights(policy_dqn.get_weights())

                    step_count=0
        # Close environment
        env.close()
        print("Reached the goal this many times: ", (rewards_per_episode > 0).sum())

        # Save policy
        with open('policy_dqn.pkl', 'wb') as file:
            pickle.dump(policy_dqn.get_weights(), file)

        print("Saving succesful")

        # Create new graph 
        plt.figure(1)

        # Plot average rewards (Y-axis) vs episodes (X-axis)
        # sum_rewards = np.zeros(episodes)
        #for x in range(episodes):
        #    sum_rewards[x] = np.sum(rewards_per_episode[max(0, x-100):(x+1)])
        
        plt.subplot(221) # plot on a 2 row x 2 col grid, at cell 1
        #plt.plot(sum_rewards)
        plt.scatter(np.arange(episodes),rewards_per_episode)
        plt.title("Rewards per episode")

        # Plot epsilon decay (Y-axis) vs episodes (X-axis)
        plt.subplot(222) # plot on a 2 row x 2 col grid, at cell 2
        plt.plot(epsilon_history)
        plt.title("Epsilon in each episode")
        
        plt.subplot(223)
        plt.plot(loss_list)
        plt.title("Loss per episode")

        plt.subplot(224)
        plt.plot(learning_rate_history)
        plt.title("Learning rate in each episode")

        # Save plots
        plt.savefig('frozen_lake_dql.png')

    # Optimize policy network
    def optimize(self, mini_batch, policy_dqn, target_dqn, learning_rate):

        # Get number of input nodes
        num_states = policy_dqn.in_features

        current_q_list = []
        target_q_list = []
        
        for state, action, new_state, reward, terminated in mini_batch:

            if terminated: 
                # Agent either reached goal (reward=1) or fell into hole (reward=0)
                # When in a terminated state, target q value should be set to the reward.
                target = reward
                
            else:
                # Calculate target q value 
                target = reward + self.discount_factor_g * target_dqn.forward(self.state_to_dqn_input(new_state, num_states)).max()
            

  
            # Get the current set of Q values
            current_q = policy_dqn.forward(self.state_to_dqn_input(state, num_states))
            current_q_list.append(current_q)
            
            # Get the target set of Q values
            target_q = target_dqn.forward(self.state_to_dqn_input(state, num_states)) # TODO? [0]

            # Adjust the specific action to the target that was just calculated
            target_q[action] = target
            target_q_list.append(target_q)
                
        # Compute loss for the whole minibatch
        loss = compute_loss(np.concatenate(target_q_list), np.concatenate(current_q_list))
        
        # Optimize the model 
        gradient = compute_gradient(np.concatenate(target_q_list), np.concatenate(current_q_list))
        

        # To save input in Neural Network
        inp = [np.array(self.state_to_dqn_input(state, num_states)) for state, action, new_state, reward, terminated in mini_batch]
        inp = np.vstack(inp)
        policy_dqn.forward(inp.T) # TODO Sketchy

        policy_dqn.backward(gradient)
        policy_dqn.update(learning_rate)

        return loss
        
   
    '''
    Converts an state (int) to a tensor representation.
    For example, the FrozenLake 4x4 map has 4x4=16 states numbered from 0 to 15. 

    Parameters: state=1, num_states=16
    Return: tensor([0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    '''
    def state_to_dqn_input(self, state:int, num_states:int):
        input_tensor = np.zeros(num_states)
        input_tensor[state] = 1
        return input_tensor

    # Run the FrozeLake environment with the learned policy
    def test(self, episodes, is_slippery=False, render = True, relu = False, hidden_layer_size = 16):
        succesful = 0

        # Create FrozenLake instance
        # env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=is_slippery, render_mode='human')
        rendered = 'human' if render else None
        env = gym.make('FrozenLake-v1', desc = ["SFFF","FFFF","FFFF","FFFG"], is_slippery=is_slippery, render_mode=rendered)

        num_states = env.observation_space.n
        num_actions = env.action_space.n

        # Load learned policy
        if (relu == True):
            policy_dqn = NN_Relu(in_states=num_states, h1_nodes=hidden_layer_size, out_actions=num_actions, batch_size = self.mini_batch_size) 
        else:
            policy_dqn = NN(in_states=num_states, h1_nodes=hidden_layer_size, out_actions=num_actions, batch_size = self.mini_batch_size) 

        
        # policy_dqn.eval()    # TODO ? switch model to evaluation mode
        with open('policy_dqn.pkl', 'rb') as file:
            policy_model = pickle.load(file)
        policy_dqn.set_weights(policy_model)
        print("Loading successful")

        # print('Policy (trained):')
        # self.print_dqn(policy_dqn)

        for _ in range(episodes):
            state = env.reset()[0]  # Initialize to state 0
            terminated = False      # True when agent falls in hole or reached goal
            truncated = False       # True when agent takes more than 200 actions            

            # Agent navigates map until it falls into a hole (terminated), reaches goal (terminated), or has taken 200 actions (truncated).
            while(not terminated and not truncated):  
                # Select best action   
                action = policy_dqn.forward(self.state_to_dqn_input(state, num_states)).argmax().item()

                # Execute action
                state,reward,terminated,truncated,_ = env.step(action)

                
        if (reward == 1):
            print("The agent reached the goal!")
            succesful = 1


        env.close()
        return succesful

    # Print DQN: state, best action, q values
    def print_dqn(self, dqn):
        # TODO NOT COMPATIBLE YET, BUT IDK IF NEEDED

        # Get number of input nodes
        num_states = dqn.fc1.in_features

        # Loop each state and print policy to console
        for s in range(num_states):
            #  Format q values for printing
            q_values = ''
            for q in dqn(self.state_to_dqn_input(s, num_states)).tolist():
                q_values += "{:+.2f}".format(q)+' '  # Concatenate q values, format to 2 decimals
            q_values=q_values.rstrip()              # Remove space at the end

            # Map the best action to L D R U
            best_action = self.ACTIONS[dqn(self.state_to_dqn_input(s, num_states)).argmax()]

            # Print policy in the format of: state, action, q values
            # The printed layout matches the FrozenLake map.
            print(f'{s:02},{best_action},[{q_values}]', end=' ')         
            if (s+1)%4==0:
                print() # Print a newline every 4 states

if __name__ == '__main__':

    # TODO target network synchronization
    # Initializing
    total_start = time.time()
    is_slippery_surface = False
    render_training = False
    render_testing = False
    relu = False
    sum_of_successes = 0
    number_of_experiments = 20
    hidden_layer_size = 16
    epoch_number = 10_000 # default: 10_000 (arbitrary)

    for i in np.arange(number_of_experiments)+1:
        print("Experiment number: ", i)
        start = time.time()
        frozen_lake = FrozenLakeDQL()
        

        # Training 
        frozen_lake.train(epoch_number, render = render_training, is_slippery=is_slippery_surface, relu = relu, hidden_layer_size = hidden_layer_size)

        # Measuring time
        end = time.time()
        print("Trained took: ", end - start)

        # Testing
        sum_of_successes += frozen_lake.test(10, is_slippery=is_slippery_surface, render = render_testing, relu = relu, hidden_layer_size= hidden_layer_size)
        print("Successes so far: ", sum_of_successes)
        print("Proportion: ", sum_of_successes/i) #TODO word
        print("")

    print("Number of successes: ", sum_of_successes)
    print("Proportion of successes: ", sum_of_successes/number_of_experiments)
    # Measuring time
    total_end = time.time()
    print("Trained took: ", total_end - total_start)


    