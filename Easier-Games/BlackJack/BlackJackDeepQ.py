import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random
import pickle
import time
import tqdm

# ---------------------------------------------------------
# Linear layer
# ---------------------------------------------------------
class Linear:
    """ A fully connected layer implemented with NumPy arrays, without any sophistication
        Uses: 
            - simple initialization of weights
            - gradient descent          
    """

    def __init__(
        self, in_features: int, out_features: int, batch_size: int
    ) -> None:
        super(Linear, self).__init__()
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed=42)
        self.weight = self.rng.normal(size=(in_features, out_features)) * np.sqrt(1.0 / in_features)
        self.bias = self.rng.normal(size=(out_features,)) * np.sqrt(1.0 / in_features)
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
# Linear layer with ADAM
# ---------------------------------------------------------
class linear_adam:
    """A fully connected layer implemented with NumPy arrays."""

    def __init__(
        self, in_features: int, out_features: int, batch_size: int
    ) -> None:
        super(linear_adam, self).__init__()
        self.batch_size = batch_size

        # For comparison: more primitive initialization of weights and bias
        # self.weight = np.random.normal(size=(in_features, out_features)) * np.sqrt(1.0 / in_features)
        # self.bias = np.random.normal(size=(out_features,)) * np.sqrt(1.0 / in_features)

        # Initialization of weights like in torch
        self.k = np.sqrt(1/in_features)
        self.rng = np.random.default_rng(seed=42)
        self.weight = self.rng.uniform(-self.k,self.k, size=(in_features, out_features))
        self.bias = self.rng.uniform(-self.k,self.k, size=(out_features,))

        # Initialization for forward and backward pass
        self.grad_weight = np.zeros((in_features, out_features))
        self.grad_bias = np.zeros(out_features)
        self.input = np.zeros((batch_size, in_features))


        # Attributes for ADAM

        # Shared for weights and bias
        self.e = 1e-8
        self.lr = 0.001
        self.b_1 = 0.9
        self.b_1_t = self.b_1
        self.b_2 = 0.999
        self.b_2_t = self.b_2

        # For weights
        self.m_w = 0
        self.v_w = 0
          
        # For bias
        self.m_b = 0
        self.v_b = 0
        
        
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

    def update(self, learning_rate) -> None:
        "Implementation of the ADAM optimizer"

        # Weight update
        self.m_w = self.b_1 * self.m_w +(1-self.b_1) * self.grad_weight / self.batch_size
        self.v_w = self.b_2 * self.v_w + (1-self.b_2) * (self.grad_weight / self.batch_size)**2  
        m_hat_w = self.m_w / (1-self.b_1_t) 
        v_hat_w = self.v_w / (1-self.b_2_t)
        self.weight  = self.weight - self.lr * m_hat_w / (np.sqrt(v_hat_w) + self.e)                    

        # Bias update
        self.m_b = self.b_1 * self.m_b +(1-self.b_1) * self.grad_bias / self.batch_size
        self.v_b = self.b_2 * self.v_b + (1-self.b_2) * (self.grad_bias / self.batch_size)**2  
        m_hat_b = self.m_b / (1-self.b_1_t) 
        v_hat_b = self.v_b / (1-self.b_2_t)
        self.bias = self.bias - self.lr * m_hat_b / (np.sqrt(v_hat_b) + self.e)   

        # Shared parameter update
        self.b_1_t = self.b_1_t*self.b_1
        self.b_2_t = self.b_2_t*self.b_2

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
        # Computes the gradient of ReLU
        grad_input = grad_output.copy()
        grad_input[self.input <=0] = 0
        return grad_input
    

# ---------------------------------------------------------
# Neural Networks
# ---------------------------------------------------------

# Define model
class three_layers_neural_network:
    def __init__(self, in_states, h1_nodes, out_actions, batch_size: int, adam: bool, relu: bool):
        super(three_layers_neural_network, self).__init__()

        self.in_features = in_states

        # Define network layers
        if(adam == True):
            
            self.l1 = linear_adam(in_states, h1_nodes, batch_size)   # Linear layer with adam optimizer
            self.l2 = linear_adam(h1_nodes, h1_nodes, batch_size)   # Linear layer with adam optimizer
            self.l3 = linear_adam(h1_nodes, out_actions, batch_size)   # Linear layer with adam optimizer
        else:
            self.l1 = Linear(in_states, h1_nodes, batch_size)   # Linear layer with gradient descent
            self.l2 = Linear(h1_nodes, h1_nodes, batch_size)   # Linear layer with gradient descent
            self.l2 = Linear(h1_nodes, out_actions, batch_size)   # Linear layer with gradient descent

        # Define activation function
        if(relu == True):
            self.a1 = Relu(h1_nodes, batch_size) # Relu activation layer
            self.a2 = Relu(h1_nodes, batch_size) # Relu activation layer
        else:
            self.a1 = Sigmoid(h1_nodes, batch_size) # Sigmoid activation layer
            self.a2 = Sigmoid(h1_nodes, batch_size) # Sigmoid activation layer


    def forward(self, x: np.ndarray) -> np.ndarray:
        
        x = np.array(x)
        x = x.reshape(x.shape[0], -1)  # Flatten the input
        x = x.T
        x = self.l1.forward(x)    # Linear layer
        x = self.a1.forward(x)    # Apply sigmoid activation function
        x = self.l2.forward(x)    # Linear layer
        x = self.a2.forward(x)    # Apply sigmoid activation function
        x = self.l3.forward(x)    # Linear layer

        return x[0]
    
    def backward(self, x: np.ndarray) -> None:
        x = self.l3.backward(x)
        x = self.a2.backward(x)
        x = self.l2.backward(x)
        x = self.a1.backward(x)
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
        self.l1.set_weights(w[0])
        self.l2.set_weights(w[1])
        self.l3.set_weights(w[2])

    def print_weights(self):
        # Prints weights of the neurons
        print("Three Layer Neural Network")
        print("First layer: ", self.l1.get_weights()) 
        print("Second layer: ", self.l2.get_weights()) 
        print("Third layer: ", self.l3.get_weights())


class two_layers_neural_network:
    def __init__(self, in_states, h1_nodes, out_actions, batch_size: int, adam: bool, relu: bool):
        super(two_layers_neural_network, self).__init__()
        self.in_features = in_states

        # Define network layers
        if(adam == True):
            
            self.l1 = linear_adam(in_states, h1_nodes, batch_size)   # Linear layer with adam optimizer
            self.l2 = linear_adam(h1_nodes, out_actions, batch_size)   # Linear layer with adam optimizer
        else:
            self.l1 = Linear(in_states, h1_nodes, batch_size)   # Linear layer with gradient descent
            self.l2 = Linear(h1_nodes, out_actions, batch_size)   # Linear layer with gradient descent

        # Define activation function
        if(relu == True):
            self.a1 = Relu(h1_nodes, batch_size) # Relu activation layer
        else:
            self.a1 = Sigmoid(h1_nodes, batch_size) # Sigmoid activation layer


    def forward(self, x: np.ndarray) -> np.ndarray:
        x = x.reshape(x.shape[0], -1)  # Flatten the input
        x = x.T 
        x = self.l1.forward(x)    # Linear layer
        x = self.a1.forward(x)    # Apply ReLU activation function
        x = self.l2.forward(x)    # Linear layer
        return x[0]
    
    def backward(self, x: np.ndarray) -> None:
        x = self.l2.backward(x)
        x = self.a1.backward(x)
        x = self.l1.backward(x)
    
    def update(self,lr) -> None:
        self.l1.update(lr)
        self.l2.update(lr)

    def get_weights(self):
        # Returns weights of the neurons
        return [self.l1.get_weights(), self.l2.get_weights()]

    def set_weights(self, w):
        # Sets weights of the neurons
        self.l1.set_weights(w[0])
        self.l2.set_weights(w[1])

    def print_weights(self):
        # Prints weights of the neurons
        print("Two Layer Neural Network")
        print("First layer: ", self.l1.get_weights()) 
        print("Second layer: ", self.l2.get_weights()) 


# ---------------------------------------------------------
# Loss and gradient of loss
# ---------------------------------------------------------

def compute_loss_mse(target: np.ndarray, prediction: np.ndarray) -> float:
    """Return MSE"""
    return np.sum((target -prediction)**2) / prediction.shape[0] 


def compute_gradient(target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    """
    Computes the gradient of the mse error.
    """
    return 2*(prediction - target) / prediction.shape[0]
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
    discount_factor_g = 0.9         # discount rate (gamma), default: 0.9  
    network_sync_rate = 10          # number of steps the agent takes before syncing the policy and target network, default: 10
    replay_memory_size = 1_000       # size of replay memory, default: 1_000
    mini_batch_size = 32          # size of the training data set sampled from the replay memory, default: 32

    # Hyperparameters which are obsolete when using ADAM
    learning_rate_a = 0.1      # learning rate (alpha), default: 0.001 (tutorial) or 0.1 (empirical)
    learning_rate_reductions = 20.0 # what part of the epochs needs to pass until we reduce the learning rate, default: 20
    learning_rate_divisor = 1.2 # the number which divides the learning rate, default: 1.2
    


    # Train the FrozeLake environment
    def train(self, episodes, render = None, relu = True, two_layers = False, adam = True, hidden_layer_size = 16):

        # Create FrozenLake instance

        
        # Creating environment
        env = gym.make('Blackjack-v1', natural=False, sab=False)
        loss_list = []   

        # Initializing constants
        num_states = 0
        for i in env.observation_space:
            num_states += i.n

        num_actions = env.action_space.n
        
        # Initializing changing variables
        lr = self.learning_rate_a
        epsilon = 1 # 1 = 100% random actions


        memory = ReplayMemory(self.replay_memory_size)

        # Create policy and target network. Number of nodes in the hidden layer can be adjusted.
        if(two_layers == True):
            policy_dqn = two_layers_neural_network(in_states=num_states, h1_nodes=hidden_layer_size, out_actions=num_actions, batch_size = self.mini_batch_size, adam = adam, relu = relu)

            target_dqn = two_layers_neural_network(in_states=num_states, h1_nodes=hidden_layer_size, out_actions=num_actions, batch_size = self.mini_batch_size, adam = adam, relu = relu)
        else: 
            policy_dqn = three_layers_neural_network(in_states=num_states, h1_nodes=hidden_layer_size, out_actions=num_actions, batch_size = self.mini_batch_size, adam = adam, relu = relu)

            target_dqn = three_layers_neural_network(in_states=num_states, h1_nodes=hidden_layer_size, out_actions=num_actions, batch_size = self.mini_batch_size, adam = adam, relu = relu)

        # Make the target and policy networks the same (copy weights and biases from one network to the other)
        target_dqn.set_weights(policy_dqn.get_weights())

        # List to keep track of rewards collected per episode. Initialize list to 0's.
        rewards_per_episode = np.zeros(episodes)

        # List to keep track of epsilon decay
        epsilon_history = []

        # List to keep track of learning rate
        learning_rate_history = []

        # Track number of steps taken. Used for syncing policy => target network.
        step_count=0

        for i in tqdm.tqdm(range(episodes)):
            # For debugging: If we print stuff during epochs, it breaks the progress bar
            # if(i%500 == 0):
            #     print("Epoch: ", i)
            
            # For comparison: a primitive learning rate scheduler
            if(adam == False):
                if(i % (episodes/self.learning_rate_reductions) == 0):  # possible augmentation: 1. constant learning rate at first and 2. learning rate reset
                    lr = lr/self.learning_rate_divisor 

            # For plotting the learning rate
            learning_rate_history.append(lr)

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

                # Keep track of the rewards collected per episode.
                rewards_per_episode[i] += reward

            
            # Check if enough experience has been collected and if at least 1 reward has been collected
            if (len(memory) > self.mini_batch_size and np.max(rewards_per_episode) > 0): 
                mini_batch = memory.sample(self.mini_batch_size)
                loss_list.append(self.optimize(mini_batch, policy_dqn, target_dqn, lr))        

                # Decay epsilon
                epsilon = max(epsilon - 1/episodes, 0.00) # possible augmentation: set a minimum epsilon (i.e. change to 0.1)

                epsilon_history.append(epsilon)

                # Copy policy network to target network after a certain number of steps
                if step_count > self.network_sync_rate:
                    target_dqn.set_weights(policy_dqn.get_weights())

                    step_count=0

        # Close environment
        env.close()

        # # Debugging logs:
        # # Log how often the agent won the game
        # print("Reached the goal this many times: ", (rewards_per_episode > 0).sum())

        # Saving the model
        with open('policy_dqn.pkl', 'wb') as file:
            pickle.dump(policy_dqn.get_weights(), file)
        

        # Create new graph 
        plt.figure(1)
        
        # Plot rewards in every episode
        plt.subplot(221) 
        plt.plot(rewards_per_episode)
        plt.title("Rewards per episode")

        # Debug log: For plotting epsilon
        # Plot epsilon decay (Y-axis) vs episodes (X-axis)
        # plt.subplot(222) # plot on a 2 row x 2 col grid, at cell 2
        # plt.plot(epsilon_history)
        # plt.title("Epsilon in each episode")
        
        # Plot average rewards (Y-axis) vs episodes (X-axis)
        plt.subplot(222)
        sum_rewards = np.zeros(episodes)
        for x in range(episodes):
           sum_rewards[x] = np.sum(rewards_per_episode[max(0, x-100):(x+1)])/((x+1)-max(0, x-100))
        plt.plot(sum_rewards)
        plt.title("Average reward")

        # Plot the loss
        plt.subplot(223)
        plt.plot(loss_list)
        plt.title("Loss per episode")

        # Plot the learning rate
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
            target_q = target_dqn.forward(self.state_to_dqn_input(state, num_states)) 

            # Adjust the specific action to the target that was just calculated
            target_q[action] = target
            target_q_list.append(target_q)
                
        # Compute loss for the whole minibatch
        loss = compute_loss_mse(np.concatenate(target_q_list), np.concatenate(current_q_list))
        
        # Optimize the model 
        gradient = compute_gradient(np.concatenate(target_q_list), np.concatenate(current_q_list))
        
        # To save input in Neural Network
        inp = [np.array(self.state_to_dqn_input(state, num_states)) for state, _, _, _, _ in mini_batch]
        inp = np.vstack(inp)
        policy_dqn.forward(inp.T) 

        policy_dqn.backward(gradient)
        policy_dqn.update(learning_rate)

        return loss

    # Different function for encoding (only one 1), the neural networks learns worse with it
    # def state_to_dqn_input(self, state, num_states:int):
    #     '''
    #     Converts an state (tuple) to a tensor representation.
        
    #     Return: tensor e.g. ([0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.])
    #     '''
        
    #     input_tensor = np.zeros(num_states)


    #     input_tensor[(state[0]-1)+32*(state[1]-1)+352*state[2]] = 1
        
    #     if(np.sum(input_tensor) != 1):
    #         raise ValueError("Method state_to_dqn_input brocken!")
       
    #     return input_tensor
   

    def state_to_dqn_input(self, state, num_states:int):
        '''
        Converts an state (tuple) to a tensor representation.
        
        Return: tensor e.g. ([0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0.])
        '''
        
        input_tensor = np.zeros(num_states)


        input_tensor[state[0]-1] = 1
        input_tensor[state[1]+31] = 1
        input_tensor[state[2]+43] = 1

        # This is to check whether the code is malfunctioning, it can be removed soon
        if(np.sum(input_tensor) != 3):
            raise ValueError("Method state_to_dqn_input brocken!")
        return input_tensor

    # Run the FrozeLake environment with the learned policy
    def test(self, episodes, render = None, relu = True, two_layers = True, adam = True, hidden_layer_size = 16):
        succesful = 0
        failure = 0


        # Create FrozenLake instance
        env = gym.make('Blackjack-v1', natural=False, sab=False)

        # Initializing
        num_states = 0
        for i in env.observation_space:
            num_states += i.n
        num_actions = env.action_space.n

        # Initialize Neural Network
        if(two_layers == True):
            policy_dqn = two_layers_neural_network(in_states=num_states, 
                                     h1_nodes=hidden_layer_size, 
                                     out_actions=num_actions, 
                                     batch_size = self.mini_batch_size, 
                                     adam = adam, 
                                     relu = relu)

        else: 
            policy_dqn = three_layers_neural_network(in_states=num_states, 
                                     h1_nodes=hidden_layer_size, 
                                     out_actions=num_actions, 
                                     batch_size = self.mini_batch_size, 
                                     adam = adam, 
                                     relu = relu)   

        # Loading the model
        with open('policy_dqn.pkl', 'rb') as file:
            policy_model = pickle.load(file)
        policy_dqn.set_weights(policy_model)
        

        # Testing
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
                # print("The agent reached the goal!")
                succesful += 1
            if (reward == -1):
                failure += 1

        
           

        # Closing the environment
        env.close()


        # Returning whether the agent fulfilled their goal
        return succesful,failure

    
if __name__ == '__main__':
    # Initializing
    render_training = None # set to 'human' to see the training on a gaming screen
    render_testing = None # set to 'human' to see the testing on a gaming screen
    test_run_number = 10_000 # How often we let it show what it learned, default: 1_000

    relu = True # set to true to change the activation function from sigmoid to relu, default: True
    two_layers = True # set to true to delete the hidden layer, default: True
    adam = True # set to true to use ADAM, default: True

    number_of_experiments = 10 # How many NNs we train
    hidden_layer_size = 16 # default: 16
    epoch_number = 1_000 # default: 1_000 

    total_start = time.time()


    sum_of_successes = 0
    sum_of_losses = 0
    

    for i in np.arange(number_of_experiments)+1:
        print("Experiment number: ", i)

        # Performance logging:
        # start = time.time()

        # Initialize training class
        frozen_lake = FrozenLakeDQL()
        
        # Training 
        frozen_lake.train(
            epoch_number, 
            render = render_training, 
            relu = relu, 
            two_layers = two_layers, 
            hidden_layer_size = hidden_layer_size,
            adam = adam)

        # Performance logging:
        # # Measuring time
        # end = time.time()
        # print("Training took: ", end - start)

        # Testing and keeping track of successes
        proportion_of_successes, proportion_of_losses = frozen_lake.test(test_run_number,
                                            render = render_testing, 
                                            relu = relu, 
                                            two_layers = two_layers, 
                                            hidden_layer_size= hidden_layer_size,
                                            adam = adam) 
        
        proportion_of_successes = proportion_of_successes/ test_run_number
        proportion_of_losses = proportion_of_losses/ test_run_number
        print("Proportion of sucesses: ", proportion_of_successes) 
        print("Proportion of losses: ", proportion_of_losses)
        print(" ")

        sum_of_successes += proportion_of_successes
        sum_of_losses += proportion_of_losses
        

    
    print("Total proportion of successes: ", sum_of_successes/number_of_experiments)
    print("Total proportion of losses: ", sum_of_losses/number_of_experiments)
    

    # Measuring time
    total_end = time.time()
    print("Training took: ", total_end - total_start)


    
