import gymnasium as gym
import numpy as cp
#import cupy as cp
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

from utils_numpy import compute_loss_mse, compute_gradient, ReplayMemory
from networks_numpy import create_network


gym.register_envs(ale_py)



# Pong Deep Q-Learning
class PongDQL():
    # Hyperparameters (adjustable)
    discount_factor_g = 0.9         # discount rate of reward (gamma), default: 0.9  
    network_sync_rate = 20_000          # number of steps the agent takes before syncing the policy and target network, default: 
    replay_memory_size = 10_000       # size of replay memory, default:
    mini_batch_size = 25       # size of the training data set sampled from the replay memory, default: 32

    # Train the Pong environment
    def train(self, episodes, render = None, model_name = "three_layers", hidden_layer_size = 16):
        
        # Creating environment
        env = gym.make(#'ALE/Breakout-v5', # Not working for unkown reason
                        'PongNoFrameskip-v4',
                        # 'Pong-v4',
                        render_mode=render, 
                        obs_type="grayscale")
        
        
        loss_list = []   

        # Initializing constants
        num_states = 80 * 80 * 4 # size of the preprocessed input
        num_actions = 3 # up, down and stay

        
        # Initializing changing variables
        epsilon = 1 # 1 = 100% random actions

        # Create replay memory
        memory = ReplayMemory(self.replay_memory_size)

        # Initializing policy and target network
        policy_dqn = create_network(in_states=num_states, h1_nodes=hidden_layer_size, out_actions=num_actions, batch_size = self.mini_batch_size, model_name=model_name)        
        target_dqn = create_network(in_states=num_states, h1_nodes=hidden_layer_size, out_actions=num_actions, batch_size = self.mini_batch_size, model_name=model_name)        
        
                   
        # Print model architecture
        policy_dqn.print_name()
        
        # Make the target and policy networks the same (copy weights and biases from one network to the other)
        target_dqn.set_weights(policy_dqn.get_weights())

        # List to keep track of rewards collected per episode. Initialize list to 0's.
        rewards_per_episode = cp.zeros(episodes)

        # List to keep track of epsilon decay
        epsilon_history = []

        # Track number of steps taken. Used for syncing policy => target network.
        network_step_count=0

        best_rewards = -100
        steps = 0

        #This is the loop of one game (21 scores)
        for i in tqdm.tqdm(range(episodes)):
            # For debugging: If we print things during epochs, it breaks the progress bar
            # if(i%500 == 0):
            #     print("Epoch: ", i)

            # Initialize frame stack at episode start
            frame_stack = deque(maxlen=4)
            
            state, _ = env.reset()  # Initialize to state 0
            processed = self.state_to_dqn_input(state)
            
            # Warmup: fill frame stack with initial frames
            for _ in range(4):
                frame_stack.append(processed)
            
            terminated = False      # True when agent reaches goal
            truncated = False       # True when steps exceed limit
            cumulative_reward = 0
            
            
            # Agent plays the game until it terminates
            
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
                episode_frames = 0
                for _ in range(4):
                    new_state, reward, terminated, truncated, _ = env.step(action + 1)
                    processed = self.state_to_dqn_input(new_state)
                    frame_stack.append(processed)
                    cumulative_reward += reward
                    network_step_count += 1             # Track number of steps taken. Used for syncing policy => target network.
                    
                    if terminated or truncated:
                        break
                
                if(steps == 500): #TODO add if reward too little
                    truncated = True 
                    steps = 0    
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
                    tup1, tup2, tup3, tup4, tup5 = memory.pop()
                    memory.append((tup1, tup2, tup3, tup4, True))

                

            
            # Keep track of highest reward
            if rewards_per_episode[i]>best_rewards:
                best_rewards = rewards_per_episode[i]

                # Debug log
                #print(f'Best rewards so far: {best_rewards}')
                
            # Check if enough experience has been collected. 
            # Here we remove checking for '0 < reward', because pong training works with rewards ranging from -20 to +5 or 10 in the beginning.
            if (len(memory) > self.mini_batch_size) :
                mini_batch = memory.sample(self.mini_batch_size)
                
                # print("mini_batch.shape: ", mini_batch[0][0].shape)
                loss_list.append(self.optimize(mini_batch, policy_dqn, target_dqn))        

                # Decay epsilon
                # epsilon = max(epsilon - 1/episodes, 0.00) # possible augmentation: set a minimum epsilon (i.e. change to 0.1)
                # We added an exponential epsilon decay here. Maybe that'll help.
                epsilon_decay_rate = 0.995
                epsilon = max(epsilon * epsilon_decay_rate, 0.01)
                
                
                epsilon_history.append(epsilon)

                # Copy policy network to target network after a certain number of steps
                if network_step_count > self.network_sync_rate:
                    target_dqn.set_weights(policy_dqn.get_weights())

                    network_step_count=0


        # Saving the model
        with open("pong_dql.pkl", 'wb') as file:
            pickle.dump(policy_dqn.get_weights(), file)
            
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

        # Debug log: For plotting epsilon
        # Plot epsilon decay (Y-axis) vs episodes (X-axis)
        # plt.subplot(222) # plot on a 2 row x 2 col grid, at cell 2
        # plt.plot(epsilon_history)
        # plt.title("Epsilon in each episode")
        
        # Plot average rewards (Y-axis) vs episodes (X-axis)
        plt.subplot(222)
        sum_rewards = cp.zeros(episodes)
        for x in range(episodes):
           sum_rewards[x] = cp.sum(rewards_per_episode[max(0, x-100):(x+1)])/((x+1)-max(0, x-100))
        plt.plot(sum_rewards)
        plt.title("Average reward")

        # Plot the loss
        plt.subplot(223)
        plt.plot(loss_list)
        plt.title("Loss per episode")

        # Save plots
        plt.savefig('pong_dql.png')
        
        

    # Optimize policy network
    def optimize(self, mini_batch, policy_dqn, target_dqn):

        # Debug log
        # print("Optimizing")

        current_q_list = []
        target_q_list = []
        
        for state, action, next_state, reward, terminated in mini_batch:

            if terminated: 
                target = reward
            else:
                # next_state is already a complete stacked state (4, 80, 80)
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
        loss = compute_loss_mse(cp.concatenate(target_q_list), cp.concatenate(current_q_list))

        # Debug log
        # print("Loss: ", loss)

        # Optimize the model 
        gradient = compute_gradient(cp.concatenate(target_q_list), cp.concatenate(current_q_list))
        
        # To save input in Neural Network

        inp = [cp.array(state) for state, _, _, _, _ in mini_batch]        
        inp = cp.stack(inp)

        policy_dqn.forward(inp) 
        policy_dqn.backward(gradient)
        policy_dqn.update()

        return loss
        
    def fusion(self, old_state, state):
        fused_state = self.state_to_dqn_input(old_state)+self.state_to_dqn_input(state)
        fused_state[fused_state == 2] = 1
        return cp.array(fused_state)

    def unstacker(self,stacker):
        return stacker.ravel()
    
    def state_to_dqn_input(self, state):

        # Debug log
        # print("Before: ", state.shape)

        # Crop the frame.
        observation_frame = state[35:195]  

        # Debug log
        # print("After 1st step: ", observation_frame.shape)

        # Downsample the frame by a factor of 2.
        observation_frame = observation_frame[::2, ::2]

        # Debug log
        # print("After 2nd step: ", observation_frame.shape)
        


        # Remove the background and apply other enhancements.
        observation_frame[observation_frame == 107] = 0  # Erase the background 
        observation_frame[observation_frame == 87] = 0  # Erase the background 

        # For catching errors in the preprocessing feature
        # if(cp.unique(observation_frame).shape[0] >4):
        #     print("Light values: ", cp.unique(observation_frame))
        #     plt.imshow(observation_frame, cmap="gray")
        #     plt.colorbar()
        #     plt.show()
        #     raise(ValueError)


        observation_frame[observation_frame != 0] = 1  # Set the items (rackets, ball) to 1.

        # Debug log 
        # if (random.random()<0.025):      
        #     # print("Light values: ", cp.unique(observation_frame))
        #     plt.imshow(observation_frame, cmap="gray")
        #     plt.colorbar()
        #     plt.show()

        # Return the preprocessed frame as a 1D floating-point array.
        # observation_frame = observation_frame.astype(float).flatten()
        
        
        
        return observation_frame


    # Run the Pong environment with the learned policy
    def test(self, episodes, render = None, model_name = "three_layers", hidden_layer_size = 16):
        print("")
        print("Starting Testing")
        print("")

        # Create Pong instance
        env = gym.make(#'ALE/Breakout-v5', # Not working for unkown reason
                        'PongNoFrameskip-v4',
                        # 'Pong-v4',
                        render_mode=render, 
                        obs_type="grayscale")
        
        # Initializing constants
        num_states = 80 * 80 * 4 # size of the preprocessed input
        num_actions = 3 # up, down and stay


        # Initialize Neural Network
        policy_dqn = create_network(in_states=num_states, h1_nodes=hidden_layer_size, out_actions=num_actions, batch_size = self.mini_batch_size, model_name=model_name)        

        # Print model architecture
        policy_dqn.print_name()

        # Loading the model
        with open("pong_dql.pkl", 'rb') as file:
            policy_model = pickle.load(file)
        policy_dqn.set_weights(policy_model)

        # Testing
        for _ in range(episodes):
            state = env.reset()[0]  # Initialize to state 0
            terminated = False      
            truncated = False  

            # Frame skipping variables
            start = True
            frames = 1 # keep track of the frame so that we now which to skip
            action = 0 # We do nothing for the first 3 skipped frames


            # Agent navigates map until the game ends
            while(not terminated and not truncated):  

                if(frames == 4):
                    fused_state = self.fusion(old_state,state)
                    if(start == True):
                        stacker = cp.stack((fused_state,fused_state,fused_state,fused_state))
                        start = False
                    else:
                        stacker[1::] = stacker[0:3:]
                        stacker[0] = fused_state  
                
                    
                    
                    # select best action   
                    # self.unstacker(stacker)
                    action = policy_dqn.forward(stacker).argmax().item()
                
                
                    # Reset frame count
                    frames = 1
                elif(frames == 3):
                    frames += 1
                    old_state = state
                else:
                    frames += 1

                
                # Execute action
                state,_,terminated,truncated,_ = env.step(action+1) # We map the actions 0-2 to 1-3

                # Debug log
                # print("Action: ", action)


        # Closing the environment
        env.close()

    
if __name__ == '__main__':
    # Initializing
    doTest = False # Set to 'True' to load and test newest model
    doTrain = True # Set to 'True' to train a new model
    
    render_training = None # set to 'human' to see the training on a gaming screen
    render_testing = 'human' # set to 'human' to see the testing on a gaming screen
    
    test_run_number = 10 # How often we let it show what it learned


    models = ["two_layers", "three_layers", "custom_model_pong", "convolutional_pong"]
    model_name = models[2]

    number_of_experiments = 1 # How many NNs we train
    hidden_layer_size = 200 # default: 
    epoch_number = 100 # default: 1_000?

    total_start = time.time()

    
    
    for i in cp.arange(number_of_experiments)+1:
        print("Experiment number: ", i)

        # Performance logging:
        # start = time.time()

        # Initialize training class
        pong = PongDQL()
        
        # Training 
        if(doTrain == True):
            pong.train(
                    episodes = epoch_number, 
                    render = render_training, 
                    model_name = model_name,
                    hidden_layer_size = hidden_layer_size,
                    )
        elif(doTest == True):
            pong.test(episodes = test_run_number,
                    render = render_testing, 
                    model_name = model_name,
                    hidden_layer_size = hidden_layer_size,
                    )



        # Performance logging:
        # Measuring time
        # end = time.time()
        # print("Training took: ", end - start)

        # Testing 



    # Measuring time
    total_end = time.time()
    print("Whole training took: ", total_end - total_start)


    
