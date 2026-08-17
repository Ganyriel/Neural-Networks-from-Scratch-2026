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

from utils_numpy import compute_loss_mse, compute_gradient, ReplayMemory
from networks_numpy import create_network


# MountainCar Deep Q-Learning
class MountainCarDQL():
    # Hyperparameters (adjustable)
    discount_factor_g = 0.9         # discount rate (gamma), default: 0.9  
    network_sync_rate = 50_000          # number of steps the agent takes before syncing the policy and target network, default: 10
    replay_memory_size = 10_000       # size of replay memory, default: 1_000
    mini_batch_size = 256        # size of the training data set sampled from the replay memory, default: 32

    
    num_divisions = 15

    # Initializing number of states
    num_states = 0
    


    # Train the MountainCar environment
    def train(self, episodes, render = None, model_name = "three_layers", hidden_layer_size = 16):

        # Create MountainCar instance

        
        # Creating environment
        env = gym.make(
            "MountainCar-v0", 
            render_mode=render            
        )
        loss_list = []   

        # Initializing constants
        num_states = env.observation_space.shape[0] # expecting 2: position & velocity
        num_actions = env.action_space.n

        # Keeping track of number of states for other functions
        self.num_states = num_states

        # Divide position and velocity into segments
        self.pos_space = np.linspace(env.observation_space.low[0], env.observation_space.high[0], self.num_divisions)    # Between -1.2 and 0.6
        self.vel_space = np.linspace(env.observation_space.low[1], env.observation_space.high[1], self.num_divisions)    # Between -0.07 and 0.07
    
        
        # Initializing changing variables
        epsilon = 1 # 1 = 100% random actions


        # Initializing Replay Memory
        memory = ReplayMemory(self.replay_memory_size)
        
        # Create policy and target network
        policy_dqn = create_network(in_states=num_states, h1_nodes=hidden_layer_size, out_actions=num_actions, batch_size = self.mini_batch_size, model_name=model_name)        
        target_dqn = create_network(in_states=num_states, h1_nodes=hidden_layer_size, out_actions=num_actions, batch_size = self.mini_batch_size, model_name=model_name)        

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
        
        best_rewards=-1_000
        highest_height = -10
        terminated_sum = 0

        for i in tqdm.tqdm(range(episodes)):
            # For debugging: If we print stuff during epochs, it breaks the progress bar
            # if(i%500 == 0):
            #     print("Epoch: ", i)
            


            state = env.reset()[0]  # Initialize to state 0
            terminated = False      # True when agent reaches goal
            truncated = False       # True when steps exceed limit


            # Agent navigates map until it reaches goal (terminated), or has taken 200 actions (truncated).
            while(not terminated and not truncated):
                # Select action based on epsilon-greedy
                if random.random() < epsilon:
                    # select random action
                    action = env.action_space.sample() # actions: 1 = left, 2=nothing, 3=right
                else:
                    # select best action   
                    action = policy_dqn.forward(self.state_to_dqn_input(state)).argmax().item()

                # Execute action
                new_state,reward,terminated,truncated,_ = env.step(action)

                # Keep track of highest height reached
                if(new_state[0]>highest_height):
                    highest_height = new_state[0]

                # Call function to set reward
                reward = self.reward_scheduler(state, action, new_state,reward,terminated,truncated)
                      
                # Keep track of the rewards collected per episode.
                rewards_per_episode[i] += reward

                # Debug log
                # print(rewards_per_episode[i])
                
                # Save experience into memory
                memory.append((state, action, new_state, reward, terminated)) 

                # Move to the next state
                state = new_state

                # Increment step counter
                step_count+=1


            # Keep track of victories
            if(terminated == True):
                terminated_sum += 1
                
                # Debug/training log
                # print("reward: ", reward)
                # print(f"reward at step {i}", rewards_per_episode[i])
            
            # Keep track of highest reward
            if rewards_per_episode[i]>best_rewards:
                best_rewards = rewards_per_episode[i]
                #print(f'Best rewards so far: {best_rewards}')
                
            # Check if enough experience has been collected (and if at least 1 reward has been collected)
            if (len(memory) > self.mini_batch_size): 
                mini_batch = memory.sample(self.mini_batch_size)
                loss_list.append(self.optimize(mini_batch, policy_dqn, target_dqn))        

                # Decay epsilon
                epsilon = max(epsilon - 1/episodes, 0.00) # possible augmentation: set a minimum epsilon (i.e. change to 0.1)

                epsilon_history.append(epsilon)

                # Copy policy network to target network after a certain number of steps
                if step_count > self.network_sync_rate:
                    target_dqn.set_weights(policy_dqn.get_weights())

                    step_count=0


        # Saving the model
        with open("mountaincar_dql.pkl", 'wb') as file:
            pickle.dump(policy_dqn.get_weights(), file)
            
        print("Best reward: ", best_rewards)
        print("Highest height: ", highest_height)
        print("Number of successes in training: ", terminated_sum)

        # Close environment
        env.close()


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

        # Save plots
        plt.savefig('mountaincar_dql.png')

    def reward_scheduler(self, state, action, new_state,reward,terminated,truncated):
        #[-0.6, -0.4]

        # At the moment not needed for the training 
        # Right rewards
        # if(new_state[0]>-0.3):
                    
        #             reward = 1
        #             if(new_state[0]>-0.2):
        #                 reward = 2
        #                 if(new_state[0]>0.1):
        #                     reward = 4
        #                     if(new_state[0]>0.3):
        #                         reward = 10

        #                         if(terminated == True):
        #                             reward = 100

        # Warning: Makes training worse
        # if(terminated == True):
        #     reward += 100

        # For testing, yields bad results
        # reward += (np.abs(state[0]+0.5))**2

        
        # Direction rewards
        if(state[1]*(action-1) > 0):
            reward += 1 
            
        return reward

    # Optimize policy network
    def optimize(self, mini_batch, policy_dqn, target_dqn):
        # print("Optimizing")
        current_q_list = []
        target_q_list = []

        # Go through batch
        for state, action, new_state, reward, terminated in mini_batch:

            if terminated: 
                # When in a terminated state, target q value should be set to the reward.
                target = reward
                
            else:
                # Calculate target q value 
                target = reward + self.discount_factor_g * target_dqn.forward(self.state_to_dqn_input(new_state)).max()
            
            # Get the current set of Q values
            current_q = policy_dqn.forward(self.state_to_dqn_input(state))
            current_q_list.append(current_q)
            
            # Get the target set of Q values
            target_q = target_dqn.forward(self.state_to_dqn_input(state)) 

            # Adjust the specific action to the target that was just calculated
            target_q[action] = target
            target_q_list.append(target_q)
                
        # Compute loss for the whole minibatch
        loss = compute_loss_mse(np.concatenate(target_q_list), np.concatenate(current_q_list))
        
        # Optimize the model 
        gradient = compute_gradient(np.concatenate(target_q_list), np.concatenate(current_q_list))
        
        # To save input in Neural Network
        inp = [np.array(self.state_to_dqn_input(state)) for state, _, _, _, _ in mini_batch]
        inp = np.stack(inp)
        policy_dqn.forward(inp) 

        policy_dqn.backward(gradient)
        policy_dqn.update()

        return loss
        
   
    # TODO not compatible with cupy
    def state_to_dqn_input(self, state):
        state_p = np.digitize(state[0], self.pos_space)
        state_v = np.digitize(state[1], self.vel_space)
        
        return np.array([state_p, state_v])

    # Run the MountainCar environment with the learned policy
    def test(self, episodes, render = None, model_name = "three_layers", hidden_layer_size = 16):
        succesful = 0

        # Create MountainCar instance
        env = gym.make('MountainCar-v0', render_mode=render)
        
        # Initializing constants
        num_states = env.observation_space.shape[0] # expecting 2: position & velocity
        num_actions = env.action_space.n

        self.pos_space = np.linspace(env.observation_space.low[0], env.observation_space.high[0], self.num_divisions)    # Between -1.2 and 0.6
        self.vel_space = np.linspace(env.observation_space.low[1], env.observation_space.high[1], self.num_divisions)    # Between -0.07 and 0.07

        # Initialize Neural Network
        policy_dqn = create_network(in_states=num_states, h1_nodes=hidden_layer_size, out_actions=num_actions, batch_size = self.mini_batch_size, model_name=model_name)        
         

        # Loading the model
        with open("mountaincar_dql.pkl", 'rb') as file:
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
                action = policy_dqn.forward(self.state_to_dqn_input(state)).argmax().item()

                # Execute action
                state,_,terminated,truncated,_ = env.step(action)

            if (terminated == True):
                succesful += 1

        # Closing the environment
        env.close()

        # Returning whether the agent fulfilled their goal
        return succesful

    
if __name__ == '__main__':
    # Initializing
    test_run_number = 100 # How often we let it show what it learned, default: 1_000
    testing_only = False # Set to 'True' to only load and test newest model
    render_training = None # set to 'human' to see the training on a gaming screen
    render_testing = None # set to 'human' to see the testing on a gaming screen

    # Choice of model
    models = ["two_layers", "three_layers"] 
    model_name = models[1]


    number_of_experiments = 1 # How many NNs we train
    hidden_layer_size = 10 # default: 20
    epoch_number = 2_000 # default: 1_000 

    total_start = time.time()

    sum_of_successes = 0
    
    for i in np.arange(number_of_experiments)+1:
        print("Experiment number: ", i)

        # Performance logging:
        # start = time.time()

        # Initialize training class
        mountain_car = MountainCarDQL()
        
        # Training 
        if(testing_only == False):
            mountain_car.train(
                epoch_number, 
                render = render_training, 
                model_name = model_name,
                hidden_layer_size = hidden_layer_size
                )

        # Performance logging:
        # # Measuring time
        # end = time.time()
        # print("Training took: ", end - start)

        # Testing and keeping track of successes
        proportion_of_successes = mountain_car.test(test_run_number,
                                            render = render_testing, 
                                            model_name = model_name,
                                            hidden_layer_size= hidden_layer_size
                                            ) 
        
        proportion_of_successes = proportion_of_successes/ test_run_number
        
        print("Proportion of sucesses: ", proportion_of_successes) 
        
        print(" ")

        sum_of_successes += proportion_of_successes
        
        

    
    print("Total proportion of successes: ", sum_of_successes/number_of_experiments)
    
    

    # Measuring time
    total_end = time.time()
    print("Training took: ", total_end - total_start)


    
