import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random
import pickle
import time
import tqdm

import sys

# setting path
sys.path.append('../shared_files')

# file imports
import utils_numpy as ut 
import layers_numpy as ly
from networks_numpy import create_network
from run_logger import RunLogger



# BlackJack Deep Q-Learning
class BlackJackDQL():
    # Hyperparameters (adjustable)
    discount_factor_g = 0.9         # discount rate (gamma), default: 0.9  
    network_sync_rate = 10          # number of steps the agent takes before syncing the policy and target network, default: 10
    replay_memory_size = 1_000       # size of replay memory, default: 1_000
    mini_batch_size = 32          # size of the training data set sampled from the replay memory, default: 32

    # Initializing number of states
    num_states = 0

    # Train the BlackJack environment
    def train(self, episodes, render = None, model_name = "three_layers", optimizer = ut.Adam, initializator = ut.xavier_initialization, activation = ly.Relu, hidden_layer_size = 16):

        # Create BlackJack instance

        
        # Creating environment
        env = gym.make('Blackjack-v1', natural=False, sab=False)
        loss_list = []   

        # Initializing constants
        num_states = 0
        for i in env.observation_space:
            num_states += i.n

        num_actions = env.action_space.n

        # Keeping track of number of states for other functions
        self.num_states = num_states
        
        # Initializing changing variables
        epsilon = 1 # 1 = 100% random actions

        # Initializing Replay Memory
        memory = ut.ReplayMemory(self.replay_memory_size)

        # Create policy and target network
        policy_dqn = create_network(in_states=num_states, h1_nodes=hidden_layer_size, out_actions=num_actions, batch_size = self.mini_batch_size, model_name=model_name, weight_initializor = initializator, activation_function = activation, optimizer = optimizer)        
        target_dqn = create_network(in_states=num_states, h1_nodes=hidden_layer_size, out_actions=num_actions, batch_size = self.mini_batch_size, model_name=model_name, weight_initializor = initializator, activation_function = activation, optimizer = optimizer)        
        
        # Print model architecture
        policy_dqn.print_name()
        
        # Make the target and policy networks the same (copy weights and biases from one network to the other)
        target_dqn.set_weights(policy_dqn.get_weights())

        # List to keep track of rewards collected per episode. Initialize list to 0's.
        rewards_per_episode = np.zeros(episodes)

        # List to keep track of epsilon decay
        epsilon_history = []


        # Track number of steps taken. Used for syncing policy => target network.
        step_count=0

        for i in tqdm.tqdm(range(episodes)):
            # For debugging: If we print stuff during epochs, it breaks the progress bar
            # if(i%500 == 0):
            #     print("Epoch: ", i)
            

            state = env.reset()[0]  # Initialize to state 0
            terminated = False      # True when agent wins, draws or loses
            truncated = False        

            # The agent plays until the game ends
            while(not terminated and not truncated):
                # Select action based on epsilon-greedy
                if random.random() < epsilon:
                    # select random action
                    action = env.action_space.sample() # actions: hit or hold
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
                loss_list.append(self.optimize(mini_batch, policy_dqn, target_dqn))        

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
        fig, ax = plt.subplots(2, 2)
    
        # Plot rewards in every episode
        ax[0, 0].plot(rewards_per_episode)
        ax[0, 0].set_title("Rewards per episode")
    
        # Plot average reward for the last 100 episodes
        sum_rewards = np.zeros(episodes)
        for x in range(episodes):
            sum_rewards[x] = np.sum(rewards_per_episode[max(0, x-100):(x+1)])/((x+1)-max(0, x-100))
        ax[0, 1].plot(sum_rewards)
        ax[0, 1].set_title("Average reward for the last 100 episodes")

        # Plot the loss
        loss_list = np.array(loss_list)
        loss_list[loss_list > 2] = 2
        ax[1, 0].plot(loss_list)
        ax[1, 0].set_title("Loss per episode")
    
        # Plot epsilon decay (Y-axis) vs episodes (X-axis)
        ax[1, 1].plot(epsilon_history)
        ax[1, 1].set_title("Epsilon in each episode")
    
        # Formatting
        fig.tight_layout(h_pad = 2, w_pad  = 2)
   
        # Save plots
        plt.savefig('black_jack_dql.png')


    # Optimize policy network
    def optimize(self, mini_batch, policy_dqn, target_dqn):

        # Get number of input nodes
        num_states =  self.num_states # policy_dqn.in_features

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
        loss = ut.compute_loss_mse(np.concatenate(target_q_list), np.concatenate(current_q_list))
        
        # Optimize the model 
        gradient = ut.compute_gradient(np.concatenate(target_q_list), np.concatenate(current_q_list))
        
        # To save input in Neural Network
        inp = [np.array(self.state_to_dqn_input(state, num_states)) for state, _, _, _, _ in mini_batch]
        inp = np.stack(inp)
        policy_dqn.forward(inp) 

        policy_dqn.backward(gradient)
        policy_dqn.update()

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

    # Run the BlackJack environment with the learned policy
    def test(self, episodes, render = None, model_name = "three_layers",  optimizer = ut.Adam, initializator = ut.xavier_initialization, activation = ly.Relu, hidden_layer_size = 16):
        succesful = 0
        failure = 0


        # Create BlackJack instance
        env = gym.make('Blackjack-v1', natural=False, sab=False)

        # Initializing
        num_states = 0
        for i in env.observation_space:
            num_states += i.n
        num_actions = env.action_space.n

        # Initialize Neural Network
        policy_dqn = create_network(in_states=num_states, h1_nodes=hidden_layer_size, out_actions=num_actions, batch_size = self.mini_batch_size, model_name=model_name, weight_initializor = initializator, optimizer = optimizer, activation_function = activation)        
         

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
    testing_only = False # Set to 'True' to only load and test newest model
    render_training = None # set to 'human' to see the training on a gaming screen
    render_testing = None # set to 'human' to see the testing on a gaming screen
    test_run_number = 1_000 # How often we let it show what it learned, default: 1_000

    # Choice of model
    models = ["one_layer", "two_layers", "three_layers"] 
    model_name = models[1]
    hidden_layer_size = 10 # default: 16

    # Choice of optimizer
    optimizers = [ut.Adam, ut.PrimitiveOptimizer] 
    optimizer_name = optimizers[0]
    
    # Choice of weight initialization
    weight_initializations = [ut.xavier_initialization, ut.old_xavier_initialization, ut.xavier_initialization_uniform, ut.xavier_initialization_normal, ut.simple_initialization, ut.kaiming_initialization] 
    initializator_name = weight_initializations[0]
    
    # Choice of activation function
    activation_functions = [ly.Relu, ly.LeakyRelu, ly.Elu, ly.Selu, ly.Sigmoid,  ly.Sigmoid2, ly.Swish, ly.Tanh, ly.Atanh, ly.Sinusoid, ly.Cosinusoid, ly.Gaussian, ly.Softplus, ly.Identity, ly.Prelu]
    activation_name = activation_functions[0]

    # Training parameters
    number_of_experiments = 3 # How many NNs we train
    epoch_number = 2_000 # default: 2_000 

    total_start = time.time()


    # Logging parameters
    sum_of_successes = 0
    sum_of_losses = 0
    

    for i in np.arange(number_of_experiments)+1:
        print("")
        print("_________________________________________________________________")
        print("Experiment number: ", i, "/", number_of_experiments)

        # Performance logging:
        # start = time.time()

        # Initialize training class
        black_jack = BlackJackDQL()
        
        # Training 
        if(testing_only == False):
            black_jack.train(
                epoch_number, 
                render = render_training, 
                model_name = model_name,
                hidden_layer_size = hidden_layer_size,
                initializator=initializator_name,
                optimizer = optimizer_name,
                activation = activation_name
                )

        # Performance logging:
        # # Measuring time
        # end = time.time()
        # print("Training took: ", end - start)

        # Testing and keeping track of successes
        proportion_of_successes, proportion_of_losses = black_jack.test(test_run_number,
                                            render = render_testing, 
                                            model_name = model_name,
                                            hidden_layer_size = hidden_layer_size,
                                            initializator=initializator_name,
                                            optimizer = optimizer_name,
                                            activation = activation_name
                                            ) 
        
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


    
