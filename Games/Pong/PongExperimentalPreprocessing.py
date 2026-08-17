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

from utils_numpy import compute_loss_mse, compute_gradient, ReplayMemory
from networks_numpy import create_network



gym.register_envs(ale_py)


# Pong Deep Q-Learning
class PongDQL():
    # Hyperparameters (adjustable)
    discount_factor_g = 0.9         # discount rate of reward (gamma), default: 0.9  
    network_sync_rate = 10_000          # number of steps the agent takes before syncing the policy and target network, default: 
    replay_memory_size = 20_000       # size of replay memory, default:
    mini_batch_size = 25        # size of the training data set sampled from the replay memory, default: 32
    size_of_velocity_memory = 8 # for calculating the velocity and direction of the ball
    # TODO SAVE ONLY POSITION OF BALL


    # Train the Pong environment
    def train(self, episodes, render = None, model_name = "three_layers", hidden_layer_size = 16):
        
        # Creating environment
        env = gym.make('PongNoFrameskip-v4',
                        # 'Pong-v4',
                        render_mode=render, 
                        obs_type="grayscale")
        
        
        loss_list = []   

        # Initializing constants
        num_states = 6 # size of the preprocessed input
        num_actions = 3 # up, down and stay

        
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
        rewards_per_episode = cp.zeros(episodes)

        # List to keep track of epsilon decay
        epsilon_history = []

        # Track number of steps taken. Used for syncing policy => target network.
        step_count=0

        training_start = False
        best_rewards = -100

        for i in tqdm.tqdm(range(episodes)):
            # For debugging: If we print stuff during epochs, it breaks the progress bar
            # if(i%500 == 0):
            #     print("Epoch: ", i)
            

            state = env.reset()[0]  # Initialize to state 0
            terminated = False      # True when agent reaches goal
            truncated = False       # True when steps exceed limit

            # Track number of steps taken. Used for terminating early
            steps = 1 

            # Saving previous states for calculation of velocity and direction of the ball
            previous_state = [state for k in range(self.size_of_velocity_memory)]


            for j in range(self.size_of_velocity_memory):
                # Until previous state memory is full, we stand still
                action =  0

                # Execute action
                new_state,reward,_,_,_ = env.step(action+1) # We map the actions 0-2 to 1-3

                # Keep track of the rewards collected per episode.
                rewards_per_episode[i] += reward

                # Fill previous state memory
                previous_state[j] = state

                # Move to the next state
                state = new_state

            # while (len(cp.nonzero(state[35:195] == 236)[0]) == 0):
            #     # Until the ball appears, we stand still
            #     action =  0

            #     # Execute action
            #     new_state,reward,_,_,_ = env.step(action+1) # We map the actions 0-2 to 1-3

            #     # Keep track of the rewards collected per episode.
            #     rewards_per_episode[i] += reward

            #     # Update previous state memory
            #     previous_state[1:] = previous_state[0:self.size_of_velocity_memory-1] 
            #     previous_state[0] = state

            #     # Move to the next state
            #     state = new_state


                
            preprocessed_new_state = self.state_to_dqn_input(new_state, previous_state[-1])

            # Agent plays until the game ends or they have taken 2_000 actions (truncated).
            while(not terminated and not truncated):

                preprocessed_state = preprocessed_new_state

                # Select action based on epsilon-greedy
                if random.random() < epsilon:
                    # select random action from modified action space
                    action =  random.randint(0, 2) 
                else:
                    # select best action   
                    action = policy_dqn.forward(preprocessed_state).argmax().item()
                    

                # Execute action
                new_state,reward,terminated,truncated,_ = env.step(action+1) # We map the actions 0-2 to 1-3


                # Debug log
                # if reward != 0:
                #     print("reward: ", reward)

                # Preprocessing new state
                preprocessed_new_state = self.state_to_dqn_input(new_state, previous_state[0])

                # Keeping track whether we gained a positive reward
                # if(0 < reward):
                training_start = True

                # Debug log
                # print("Action taken: ", action)
                      
                # Keep track of the rewards collected per episode.
                rewards_per_episode[i] += reward

                # Debug log
                # print(rewards_per_episode[i])
                
                # Save experience into memory
                memory.append((preprocessed_state, action, preprocessed_new_state, reward, terminated)) 

                # Update previous state memory
                previous_state[1:] = previous_state[0:self.size_of_velocity_memory-1] 
                previous_state[0] = state

                # Move to the next state
                state = new_state

                if(steps == 2_000): # TODO add if reward too little
                    truncated = True 
                    steps = 0
                
                # Increment step counter
                step_count+=1
                steps += 1
            
            # Keep track of highest reward
            if rewards_per_episode[i]>best_rewards:
                best_rewards = rewards_per_episode[i]

                # Debug log
                #print(f'Best rewards so far: {best_rewards}')
                
            # Check if enough experience has been collected (and if at least 1 reward has been collected)
            if (len(memory) > self.mini_batch_size and training_start == True) :
                mini_batch = memory.sample(self.mini_batch_size)
                loss_list.append(self.optimize(mini_batch, policy_dqn, target_dqn))        

                # Decay epsilon
                epsilon = max(epsilon - 1/episodes, 0.00) # possible augmentation: set a minimum epsilon (i.e. change to 0.1)

                epsilon_history.append(epsilon)

                # Copy policy network to target network after a certain number of steps
                if step_count > self.network_sync_rate:
                    target_dqn.set_weights(policy_dqn.get_weights())

                    step_count=0

            # print("Rewards in this episode: ", rewards_per_episode[i])
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
        current_q_list = []
        target_q_list = []
        
        for preprocessed_state, action, preprocessed_new_state, reward, terminated in mini_batch:

            if reward != 0: #terminated: 
                # When in a terminated state, target q value should be set to the reward.
                target = reward
                
            else:
                # Calculate target q value 
                target = reward + self.discount_factor_g * target_dqn.forward(preprocessed_new_state).max()


            # Get the current set of Q values
            current_q = policy_dqn.forward(preprocessed_state)
            current_q_list.append(current_q)
            
            # Get the target set of Q values
            target_q = target_dqn.forward(preprocessed_state) 

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
        inp = [preprocessed_state for preprocessed_state, _, _, _, _ in mini_batch]
        inp = cp.stack(inp)
        policy_dqn.forward(inp) 

        policy_dqn.backward(gradient)
        policy_dqn.update()

        return loss
        
    def get_positions(self, state):
        # Crop the frame
        observation_frame = state[34:193]

        # Paddle position
        paddle_pixels = cp.nonzero(observation_frame == 147)
        num_paddle = len(paddle_pixels[0])

        # When starting the environment, the colours a bit different
        if num_paddle == 0:
            paddle_pixels = cp.nonzero(observation_frame == 121)
            num_paddle = len(paddle_pixels[0])
        
        

        if num_paddle == 0:
            
            plt.imshow(observation_frame, cmap="gray")
            plt.colorbar()
            plt.show()

            plt.imshow(state, cmap="gray")
            plt.colorbar()
            plt.show()


        paddle_middle = int(cp.floor(num_paddle/2))
        paddle_pos = (paddle_pixels[0][paddle_middle])
        
        # Ball position
        ball_pixels = cp.nonzero(observation_frame == 236)
        num_ball = len(ball_pixels[0])
        #print(num_ball)
        if(num_ball == 0):
            ball_pos = (200,200)
        else:
            ball_middle = int(cp.floor(num_ball/2))
            ball_pos = (ball_pixels[0][ball_middle],ball_pixels[0][ball_middle])

        return (paddle_pos, ball_pos)

    
    def state_to_dqn_input(self, state, prev_state):
        # Performs preprocessing steps

        # Debug log
        # print("Before: ", state.shape)

             
        # Positions of paddle and ball
        paddle_pos, ball_pos = self.get_positions(state)
        _, prev_ball_pos = self.get_positions(prev_state)

        if(ball_pos == (200,200) or prev_ball_pos == (200,200)):
            ball_absolute_velocity = 0
            ball_direction = cp.array([0,0])
        else:
            ball_directed_velocity = cp.array(ball_pos)-cp.array(prev_ball_pos)
            ball_absolute_velocity = cp.linalg.norm(ball_directed_velocity)
            if (ball_absolute_velocity == 0):
                ball_direction = cp.array([0,0])
            else:
                ball_direction = ball_directed_velocity/ball_absolute_velocity

                # rounding
                # ball_direction = cp.floor(ball_direction*10)/10

        # No ball
        # if (ball_direction[0] != 0 or ball_direction[1] != 0):
        #     print(ball_direction)
        # else:
        #     if (random.random()<0.01):
        #         plt.imshow(state[34:193], cmap="gray")
        #         plt.colorbar()
        #         plt.show()

        #         plt.imshow(state, cmap="gray")
        #         plt.colorbar()
        #         plt.show()

                

        return paddle_pos, ball_pos[0], ball_pos[1], ball_absolute_velocity, ball_direction[0], ball_direction[1]



        
        

        # # Debug log 
        # if (random.random()<1):      
        #     print("Light values: ", cp.unique(observation_frame))
        #     # print(observation_frame[observation_frame == 147].shape)
        #     # print(len(cp.where(observation_frame == 147)))
        #     # print(cp.where(observation_frame == 147)[0].shape)
        #     # print(cp.where(observation_frame == 147)[1].shape)
        #     # print(cp.where(observation_frame == 147))
        
        #     plt.imshow(observation_frame, cmap="gray")
        #     plt.colorbar()
        #     plt.show()
        #     print()

        


    # Run the Pong environment with the learned policy
    def test(self, episodes, render = None, model_name = "three_layers", hidden_layer_size = 16):
        # Create Pong instance
        env = gym.make('PongNoFrameskip-v4',
                        render_mode=render, 
                        obs_type="grayscale")
        
        # Initializing constants
        num_states = 6 # size of the preprocessed input
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

            # Track number of steps taken. Used for terminating early
            steps = 1 

            previous_state = [state for _ in range(self.size_of_velocity_memory)]
            
            for j in range(4):
                # Until previous state memory is full, we stand still
                action =  0
            
                # Execute action
                new_state,_,_,_,_ = env.step(action+1) # We map the actions 0-2 to 1-3
            
                # Fill previous state memory
                previous_state[j] = state
            
                # Move to the next state
                state = new_state
            
            while (len(cp.nonzero(state[35:195] == 236)[0]) == 0):
                # Until the ball appears, we stand still
                action =  0
            
                # Execute action
                new_state,_,_,_,_ = env.step(action+1) # We map the actions 0-2 to 1-3
            
                # Update previous state memory
                previous_state[1:] = previous_state[0:self.size_of_velocity_memory-1] 
                previous_state[0] = state
            
                # Move to the next state
                state = new_state
                print("No ball")
            

            # Agent navigates map until it falls into a hole (terminated), reaches goal (terminated), or has taken 200 actions (truncated).
            while(not terminated and not truncated):  

                # Preprocess state
                preprocessed_frame = self.state_to_dqn_input(state,previous_state[-1])

                # Debug log
                # print("State: ", preprocessed_frame)

                # Select best action   
                action = policy_dqn.forward(preprocessed_frame).argmax().item()
                
                # Debug log
                # print("Chosen action: ", action)

                # Execute action
                state,_,terminated,truncated,_ = env.step(action+1) # We map the actions 0-2 to 1-3

                if(steps == 1_200):
                    truncated = True 
                    steps = 1

                # Update previous state memory
                previous_state[1:] = previous_state[0:self.size_of_velocity_memory-1] 
                previous_state[0] = state
                
                # Increment step counter
                steps += 1
  

        # Closing the environment
        env.close()

    
if __name__ == '__main__':
    # Initializing
    testing_only = False # Set to 'True' to only load and test newest model
    render_training = None # set to 'human' to see the training on a gaming screen
    render_testing = 'human' # set to 'human' to see the testing on a gaming screen
    test_run_number = 10 # How often we let it show what it learned

    # Choice of model
    models = ["two_layers", "three_layers"] 
    model_name = models[0]


    number_of_experiments = 1 # How many NNs we train
    hidden_layer_size = 200 # default: 
    epoch_number = 1_000 # default: 

    total_start = time.time()

    
    
    for i in cp.arange(number_of_experiments)+1:
        print("Experiment number: ", i)

        # Performance logging:
        # start = time.time()

        # Initialize training class
        pong = PongDQL()
        
        # Training 
        if(testing_only == False):
            pong.train(
                epoch_number, 
                render = render_training, 
                model_name = model_name,
                hidden_layer_size = hidden_layer_size)

        # Performance logging:
        # # Measuring time
        # end = time.time()
        # print("Training took: ", end - start)

        # Testing 
        pong.test(test_run_number,
                    render = render_testing, 
                    model_name = model_name,
                    hidden_layer_size= hidden_layer_size) 

        
        
        print(" ")


    # Measuring time
    total_end = time.time()
    print("Training took: ", total_end - total_start)


    
