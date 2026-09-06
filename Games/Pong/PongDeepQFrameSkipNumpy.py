import gymnasium as gym
import numpy as cp
import matplotlib.pyplot as plt
from collections import deque
import random
import pickle
import time
import tqdm
import ale_py

import sys

# setting path
sys.path.append('../shared_files')

import layers_numpy as ly
import utils_numpy as ut
from networks_numpy import create_network


gym.register_envs(ale_py)



# Pong Deep Q-Learning
class PongDQL():
    # Hyperparameters (
    discount_factor_g = 0.9         # discount rate of reward (gamma), default: 0.9  
    network_sync_rate = 20_000          # number of steps the agent takes before syncing the policy and target network, default: 20_000
    replay_memory_size = 10_000       # size of replay memory, default: 10_000
    mini_batch_size = 32       # size of the training data set sampled from the replay memory, default: 32
    
    
    # Train the Pong environment
    def train(self, episodes, render = None, model_name = "three_layers", optimizer = ut.Adam, initializator = ut.xavier_initialization, activation = ly.Relu, hidden_layer_size = 16):
        
        # Creating environment
        env = gym.make('PongNoFrameskip-v4',
                        render_mode=render, 
                        obs_type="grayscale")
    

        # Steps-Based Linear epsilon decay (Idea: decay over first 1Mil. steps)
        epsilon_start = 1.0
        epsilon_end = 0.01
        epsilon_decay_steps = 1_000_000

        # ========== INITIALIZATIONS  ==========
        epsilon = epsilon_start          # Start at 100% exploration
        epsilon_history = []             # List to keep track of epsilon decay
        loss_list = []   
        
        # Initializing constants
        num_states = 80 * 80 * 4 # size of the preprocessed input
        num_actions = 3 # up, down and stay

        # Create replay memory
        memory = ut.ReplayMemory(self.replay_memory_size)

        # Initializing policy and target network
        policy_dqn = create_network(in_states=num_states, h1_nodes=hidden_layer_size, out_actions=num_actions, batch_size = self.mini_batch_size, model_name=model_name, weight_initializor = initializator, optimizer = optimizer, activation_function = activation)        
        target_dqn = create_network(in_states=num_states, h1_nodes=hidden_layer_size, out_actions=num_actions, batch_size = self.mini_batch_size, model_name=model_name, weight_initializor = initializator, optimizer = optimizer, activation_function = activation)        
        
        # Print model architecture
        policy_dqn.print_name()
        
        # Make the target and policy networks the same (copy weights and biases from one network to the other)
        target_dqn.set_weights(policy_dqn.get_weights())

        # List to keep track of rewards collected per episode. Initialize list to 0's.
        rewards_per_episode = cp.zeros(episodes)

        # Track number of steps taken. Used for syncing policy => target network, epsilon decay and truncation
        network_step_count = 0
        epsilon_steps = 0
        steps = 0
       
        # Track number of best reward
        best_rewards = -100
        
        # Loop of one game (21 scores)
        for i in tqdm.tqdm(range(episodes)):

            # Initialize frame stack at episode start
            frame_stack = deque(maxlen=4)
            
            state = env.reset()[0]  # Initialize to state 0
            processed = self.state_to_dqn_input(state)
            
            # Warmup: fill frame stack with initial frames
            for _ in range(4):
                frame_stack.append(processed)
            
            terminated = False      # True when agent reaches goal
            truncated = False       # True when steps exceed limit
            cumulative_reward = 0   # Keeping track of earned rewards
            
            
            #This is the Loop for one round
            while(not terminated and not truncated):
                # Get current stacked state
                stacked_state = cp.stack(list(frame_stack))  # Shape: (4, 80, 80)
                
             # Select action based on epsilon-greedy
                if random.random() < epsilon:
                    action = random.randint(0, 2)  # Exploration   
                else:
                    action = policy_dqn.forward(stacked_state).argmax().item()  # Exploitation
                    
                # Execute action and repeat it for 4 frames
                for _ in range(4):
                    new_state, reward, terminated, truncated, _ = env.step(action + 1)
                    processed = self.state_to_dqn_input(new_state)
                    frame_stack.append(processed)
                    cumulative_reward += reward

                    # Track number of steps taken. Used for syncing policy => target network and epsilon decay
                    network_step_count += 1            
                    epsilon_steps +=1

                    # Break if the game is supposed to end
                    if terminated or truncated:
                        break

                # Truncated after 500 x 4 steps where taken
                if(steps == 500): 
                    truncated = True 
                    steps = 0  

                # Keep track of steps taken
                steps += 1
                
                # Keep track of rewards collected per episode
                rewards_per_episode[i] += reward
                
                # Get new stacked state after action
                next_stacked_state = cp.stack(list(frame_stack))
                
                # Save experience into memory
                memory.append((stacked_state, action, next_stacked_state, cumulative_reward, terminated))
                
                # Reset cumulative reward
                cumulative_reward = 0
                
                if(terminated == True):
                    # Mark the last transition as terminal
                    tup1, tup2, tup3, tup4, _ = memory.pop()
                    memory.append((tup1, tup2, tup3, tup4, True))

            
            # Keep track of highest reward
            if rewards_per_episode[i] > best_rewards:
                best_rewards = rewards_per_episode[i]

                
            # Check if enough experience has been collected
            if (len(memory) > self.mini_batch_size) :
                mini_batch = memory.sample(self.mini_batch_size)
                
                # print("mini_batch.shape: ", mini_batch[0][0].shape)
                loss_list.append(self.optimize(mini_batch, policy_dqn, target_dqn))        

                epsilon = max(
                    epsilon_end,
                    epsilon_start - (epsilon_start - epsilon_end) * (epsilon_steps / epsilon_decay_steps)
                )
                  
                epsilon_history.append(epsilon)

                # Copy policy network to target network after a certain number of steps
                if network_step_count > self.network_sync_rate:
                    target_dqn.set_weights(policy_dqn.get_weights())

                    # Reset steps
                    network_step_count = 0


        # Saving the model
        with open("pong_dql.pkl", 'wb') as file:
            pickle.dump(policy_dqn.get_weights(), file)

        # Print best reward
        print("Best reward: ", best_rewards)
        
        # Close environment
        env.close()
        
        # Plotting
        # Create new graph 
        plt.figure(1)
        
        # Plot rewards in every episode
        plt.subplot(221) 
        plt.plot(rewards_per_episode)
        plt.title("Rewards per episode")

        # Plot epsilon decay 
        plt.subplot(222) 
        plt.plot(epsilon_history)
        plt.title("Epsilon in each episode")
        
        # Plot the loss
        plt.subplot(223)
        plt.plot(loss_list)
        plt.title("Loss per episode")
        
        # Plot average rewards from the last 100 episodes
        sum_rewards = cp.zeros(episodes)
        for x in range(episodes):
           sum_rewards[x] = cp.sum(rewards_per_episode[max(0, x-100):(x+1)])/((x+1)-max(0, x-100))

        plt.subplot(224)
        plt.plot(sum_rewards, linewidth=2, color='blue', label='Agent Performance')

        # Add horizontal reference lines
        plt.axhline(y=-21, color='red', linestyle='--', alpha=0.5, linewidth=1, 
                    label='Random Agent Baseline (-21)')
        plt.axhline(y=0, color='green', linestyle='--', alpha=0.5, linewidth=1,
                    label='Learning Threshold (0)')
        plt.title("Performance vs. Baselines", fontsize=12)
        plt.xlabel("Episode", fontsize=10)
        plt.ylabel("Reward", fontsize=10)
        plt.legend(loc='best', fontsize=8)
        plt.grid(True, alpha=0.3)
        plt.ylim([-21, 5])

        # Improve layout
        plt.tight_layout(pad=2.5)
        
        # Save plots
        plt.savefig('pong_dql.png')
        
        

    # Optimize policy network
    def optimize(self, mini_batch, policy_dqn, target_dqn):

        # Initialize q lists
        current_q_list = []
        target_q_list = []
        
        for state, action, next_state, reward, terminated in mini_batch:
            # Calculate expected reward
            if terminated: 
                target = reward
            else:
                target = reward + self.discount_factor_g * target_dqn.forward(next_state).max()
            
            # Get the current set of Q values from current stacked state
            current_q = policy_dqn.forward(state)
            current_q_list.append(current_q)

            # Get the target Q values from current state (to modify only the chosen action)
            target_q = target_dqn.forward(state) 

            # Adjust the specific action to the target that was just calculated
            target_q[action] = target
            target_q_list.append(target_q)

        # Compute loss for the whole minibatch
        loss = ut.compute_loss_mse(cp.concatenate(target_q_list), cp.concatenate(current_q_list))

        # Optimize the model 
        gradient = ut.compute_gradient(cp.concatenate(target_q_list), cp.concatenate(current_q_list))
        
        # Putting input into a minibatch
        inp = [cp.array(state) for state, _, _, _, _ in mini_batch]        
        inp = cp.stack(inp)

        # Peforming forward pass
        policy_dqn.forward(inp)

        # Performing backward pass 
        policy_dqn.backward(gradient)

        # Performing weight and bias update
        policy_dqn.update()


        # Returning the loss
        return loss
        
    
    def state_to_dqn_input(self, state):
        # Preprocesses frame

        # Crop the frame.
        observation_frame = state[35:195]  

        # Downsample the frame by a factor of 2.
        observation_frame = observation_frame[::2, ::2]        


        # Remove the background and apply other enhancements.
        observation_frame[observation_frame == 107] = 0  # Erase the background 
        observation_frame[observation_frame == 87] = 0  # Erase the background 

        # Set the items (paddles, ball) to 1.
        observation_frame[observation_frame != 0] = 1  
    
        # Return preprocessed frame
        return observation_frame


    # Run the Pong environment with the learned policy
    def test(self, episodes, render = None, model_name = "three_layers", optimizer = ut.Adam, initializator = ut.xavier_initialization, activation = ly.Relu, hidden_layer_size = 16):
        print("")
        print("Starting Testing")
        print("")

        # Create Pong instance
        env = gym.make('PongNoFrameskip-v4',
                        render_mode=render, 
                        obs_type="grayscale")
        
        # Initializing constants
        num_states = 80 * 80 * 4 # size of the preprocessed input
        num_actions = 3 # up, down and stay

        # Initialize Neural Network
        policy_dqn = create_network(in_states=num_states, h1_nodes=hidden_layer_size, out_actions=num_actions, batch_size = self.mini_batch_size, model_name=model_name, weight_initializor = initializator, optimizer = optimizer, activation_function = activation)        

        # Print model architecture
        policy_dqn.print_name()

        # Loading the model
        with open("pong_dql.pkl", 'rb') as file:
            policy_model = pickle.load(file)
        policy_dqn.set_weights(policy_model)

        # Testing
        for _ in range(episodes):
            # Initialize frame stack at episode start
            frame_stack = deque(maxlen=4)
            
            state = env.reset()[0]  # Initialize to state 0
            processed = self.state_to_dqn_input(state)
            
            # Warmup: fill frame stack with initial frames
            for _ in range(4):
                frame_stack.append(processed)
            
            terminated = False      # True when agent reaches goal
            truncated = False       # True when steps exceed limit

            # Agent plays game until it terminates
            while(not terminated and not truncated):
                # Get current stacked state
                stacked_state = cp.stack(list(frame_stack))  # Shape: (4, 80, 80)

                # Perform action
                action = policy_dqn.forward(stacked_state).argmax().item()
                                
                # Execute action and repeat it for 4 frames
                for _ in range(4):
                    new_state, _, terminated, _, _ = env.step(action + 1)
                    processed = self.state_to_dqn_input(new_state)
                    frame_stack.append(processed)
                   
                    # Break if the game is supposed to end
                    if terminated:
                        break
            

        # Closing the environment
        env.close()

    
if __name__ == '__main__':
    # Initializing
    doTest = True # Set to 'True' to load and test newest model
    doTrain = True # Set to 'True' to train a new model
    
    render_training = None # set to 'human' to see the training on a gaming screen
    render_testing = 'human' # set to 'human' to see the testing on a gaming screen
    
    test_run_number = 10 # How often we let it show what it learned

    # Choice of model
    models = ["two_layers", "three_layers", "triple_convolutional_model_pong", "convolutional_pong"]
    model_name = models[2]

    # Choice of optimizer
    optimizers = [ut.Adam, ut.PrimitiveOptimizer] 
    optimizer_name = optimizers[0]
    
    # Choice of weight initialization (for linear layers only, Conv has its own initialization)
    weight_initializations = [ut.xavier_initialization, ut.old_xavier_initialization, ut.xavier_initialization_uniform, ut.xavier_initialization_normal, ut.simple_initialization, ut.kaiming_initialization] 
    initializator_name = weight_initializations[1]
    
    # Choice of activation function
    activation_functions = [ly.Relu, ly.LeakyRelu, ly.Elu, ly.Selu, ly.Sigmoid,  ly.Sigmoid2, ly.Swish, ly.Tanh, ly.Atanh, ly.Sinusoid, ly.Cosinusoid, ly.Gaussian, ly.Softplus, ly.Identity, ly.Prelu]
    activation_name = activation_functions[0]
   
    # Parameters for training
    number_of_experiments = 1 # How many NNs we train
    hidden_layer_size = 200 # default: 
    epoch_number = 1_000 # default: 1_000?

    # For measuring time
    total_start = time.time()
    
    for i in cp.arange(number_of_experiments)+1:
        print("Experiment number: ", i)

        # Initialize training class
        pong = PongDQL()
        
        # Training 
        if(doTrain == True):
            pong.train(
                    episodes = epoch_number, 
                    render = render_training, 
                    model_name = model_name,
                    initializator=initializator_name,
                    optimizer = optimizer_name,
                    hidden_layer_size = hidden_layer_size,
                    activation = activation_name
                    )
        if(doTest == True):
            pong.test(episodes = test_run_number,
                    render = render_testing, 
                    model_name = model_name,
                    initializator=initializator_name,
                    optimizer = optimizer_name,
                    hidden_layer_size = hidden_layer_size,
                    activation = activation_name
                    )


    # Measuring time
    total_end = time.time()
    print("Whole training took: ", total_end - total_start)


    
