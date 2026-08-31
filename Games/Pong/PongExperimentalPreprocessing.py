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
    # Hyperparameters (adjustable)
    discount_factor_g = 0.9         # discount rate of reward (gamma), default: 0.9  
    network_sync_rate = 50_000          # number of steps the agent takes before syncing the policy and target network, default: 
    replay_memory_size = 20_000       # size of replay memory, default:
    mini_batch_size = 128        # size of the training data set sampled from the replay memory, default: 32
    size_of_velocity_memory = 6 # for calculating the velocity and direction of the ball
    # TODO MAYBE SAVE ONLY POSITION OF BALL


    # Train the Pong environment
    def train(self, episodes, render = None, model_name = "three_layers",  optimizer = ut.Adam, initializator = ut.xavier_initialization, activation = ly.Relu, hidden_layer_size = 16):
        
        # Creating environment
        env = gym.make('PongNoFrameskip-v4',
                        render_mode=render, 
                        obs_type="grayscale")
        
        
        loss_list = []   

        # Initializing constants
        num_states = 5 #580 # 5 # size of the preprocessed input (position of paddle, position of ball x2, velocity of ball x2)
        num_actions = 2 # size of action space (stay, up, down) 

        
        # Initializing changing variables
        epsilon = 1 # 1 = 100% random actions

        # Initializing Replay Memory
        memory = ut.ReplayMemory(self.replay_memory_size)

        # Create policy and target network
        policy_dqn = create_network(in_states=num_states, h1_nodes=hidden_layer_size, out_actions=num_actions, batch_size = self.mini_batch_size, model_name=model_name, weight_initializor = initializator, optimizer = optimizer, activation_function = activation)        
        target_dqn = create_network(in_states=num_states, h1_nodes=hidden_layer_size, out_actions=num_actions, batch_size = self.mini_batch_size, model_name=model_name, weight_initializor = initializator, optimizer = optimizer, activation_function = activation)        
                
        # Print model architecture
        policy_dqn.print_name()
        
        # Make the target and policy networks the same (copy weights and biases from one network to the other)
        target_dqn.set_weights(policy_dqn.get_weights())

        # List to keep track of rewards collected per episode. Initialize list to 0's.
        rewards_per_episode = cp.zeros(episodes)

        # List to keep track of epsilon decay
        epsilon_history = []

        # List to monitor time of an action
        monitor_time = []

        # Track number of steps taken. Used for syncing policy => target network.
        network_step_count=0

        training_start = False
        best_rewards = -100

        for i in tqdm.tqdm(range(episodes)):
            state = env.reset()[0]  # Initialize to state 0
            terminated = False      # True when agent reaches goal
            truncated = False       # True when steps exceed limit

            # Track number of steps taken. Used for terminating early
            steps_truncation = 1 

            # Saving previous states for calculation of velocity and direction of the ball
            previous_state = deque(maxlen=self.size_of_velocity_memory)

            for _ in range(self.size_of_velocity_memory):
                # Until previous state memory is full, we stand still
                action =  0

                # Execute action
                new_state,reward,_,_,_ = env.step(action+2) # We map the actions 0-2 to 1-3

                # Keep track of the rewards collected per episode.
                rewards_per_episode[i] += reward

                # Fill previous state memory
                previous_state.append(state)

                # Move to the next state
                state = new_state

            # while (len(cp.nonzero(state[35:195] == 236)[0]) == 0):
            #     # Until the ball appears, we stand still
            #     action =  0

            #     # Execute action
            #     new_state,reward,_,_,_ = env.step(action+2) # We map the actions 0-2 to 1-3

            #     # Keep track of the rewards collected per episode.
            #     rewards_per_episode[i] += reward

            #     # Update previous state memory
            #     previous_state.append(state)

            #     # Move to the next state
            #     state = new_state


             
            preprocessed_new_state = self.state_to_dqn_input(new_state, previous_state[0])

            # Agent plays until the game ends or they have taken 1_200 actions (truncated).
            while(not terminated and not truncated):

                preprocessed_state = preprocessed_new_state

                # Select action based on epsilon-greedy
                if random.random() < epsilon:
                    # select random action from modified action space
                    action =  random.randint(0, 1) 
                else:
                    # select best action   
                    action = policy_dqn.forward(preprocessed_state).argmax().item()
                    

                # Execute action
                new_state,reward,terminated,truncated,_ = env.step(action+2) # We map the actions 0-2 to 1-3


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
                if(preprocessed_state[3] == 0 and preprocessed_state[4] == 0):
                    # If no ball, save only some transitions
                    if(random.random()<0.2):
                        memory.append((preprocessed_state, action, preprocessed_new_state, reward, terminated)) 
                elif(preprocessed_state[3] <= 0):
                    # If ball is flying left, save only some transitions
                    if(random.random()<0.1):
                        memory.append((preprocessed_state, action, preprocessed_new_state, reward, terminated)) 
                else:
                    memory.append((preprocessed_state, action, preprocessed_new_state, reward, terminated)) 

                # Update previous state memory
                previous_state.append(state)

                # Move to the next state
                state = new_state

                if(steps_truncation == 1_200): # TODO add if reward too little
                    truncated = True 
                    steps_truncation = 0
                
                # Increment step counter
                network_step_count+=1
                steps_truncation += 1
            
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

                # Keep track of epsilon for plotting
                epsilon_history.append(epsilon)
                
                # Copy policy network to target network after a certain number of steps
                if network_step_count > self.network_sync_rate:
                    target_dqn.set_weights(policy_dqn.get_weights())
                    network_step_count=0
                


            # print("Rewards in this episode: ", rewards_per_episode[i])
        # Saving the model
        with open("pong_experimental_dql.pkl", 'wb') as file:
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
        
        # Plot time it takes to sample
        plt.subplot(222)        
        plt.plot(monitor_time)
        plt.title("Monitored action's time")
        
        # Plot the loss
        loss_list = cp.array(loss_list)
        loss_list[loss_list > 2] = 2
        plt.subplot(223)
        plt.plot(loss_list)
        plt.title("Loss per episode")

        # Plot epsilon decay (Y-axis) vs episodes (X-axis)
        plt.subplot(224) # plot on a 2 row x 2 col grid, at cell 2
        plt.plot(epsilon_history)
        plt.title("Epsilon in each episode")
        
        # Save plots
        plt.savefig('pong_experimental_dql.png')

        # Prints the average reward of the last 100 games
        print("Average reward in the last 100 games: ", cp.sum(rewards_per_episode[-101:-1:])/100)
        
        

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
        loss = ut.compute_loss_mse(cp.concatenate(target_q_list), cp.concatenate(current_q_list))

        # Debug log
        # print("Loss: ", loss)

        # Optimize the model 
        gradient = ut.compute_gradient(cp.concatenate(target_q_list), cp.concatenate(current_q_list))
        
        # To save input in Neural Network
        inp = [preprocessed_state for preprocessed_state, _, _, _, _ in mini_batch]
        inp = cp.stack(inp)
        # st = time.time()
        policy_dqn.forward(inp) 
        # print("Forward pass time: ", time.time()-st)

        # st = time.time()
        policy_dqn.backward(gradient)
        # print("Backward pass time: ", time.time()-st)

        # st = time.time()
        policy_dqn.update()
        # print("Update time: ", time.time()-st)
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
            ball_pos = (ball_pixels[1][ball_middle],ball_pixels[0][ball_middle])

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
            ball_directed_velocity = cp.array([0,0])
            #ball_direction = cp.array([0,0])
        else:
            ball_directed_velocity = cp.array(ball_pos)-cp.array(prev_ball_pos)
            ball_absolute_velocity = cp.linalg.norm(ball_directed_velocity)
            if (ball_absolute_velocity == 0):
                #ball_direction = cp.array([0,0])
                ball_directed_velocity = cp.array([0,0])
            # else:
            #     ball_direction = ball_directed_velocity/ball_absolute_velocity

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
        
        # Debug log
        # if (random.random()<0.01):
        #     print("Paddle pos: ", paddle_pos)
        #     print("Ball previous position: ", prev_ball_pos[0], prev_ball_pos[1])
        #     print("Ball position: ", ball_pos[0], ball_pos[1])
        #     print("Ball Velocity: ", ball_directed_velocity[0], ball_directed_velocity[1])
        #     print("")
        #     print("")
        #     plt.figure(1)
        
        #     # Plot rewards in every episode
        #     plt.subplot(221)
        #     plt.imshow(prev_state[34:193], cmap="gray")
        #     plt.colorbar()

        #     plt.subplot(222)
        #     plt.imshow(state[34:193], cmap="gray")
        #     plt.colorbar()
        #     plt.show()

        # if (random.random()<0.01):
        #     print("Ball Velocity: ", ball_directed_velocity[0], ball_directed_velocity[1])

        preprocessed_state = paddle_pos, ball_pos[0], ball_pos[1], ball_directed_velocity[0], ball_directed_velocity[1]
        # preprocessed_state = self.encode_state(paddle_pos, ball_pos[0], ball_pos[1], ball_directed_velocity[0], ball_directed_velocity[1])
        return  preprocessed_state



    def encode_state(self, paddle_pos, ball_x, ball_y, ball_directed_velocity_x, ball_directed_velocity_y):
        # 160
        # 160
        # 160
        # 50
        # 50
        vector_paddle_pos = cp.zeros(160)
        vector_paddle_pos[paddle_pos] = 1

        vector_ball_x = cp.zeros(160)
        if(ball_x != 200):
            vector_ball_x[ball_x] = 1

        vector_ball_y = cp.zeros(160)
        if(ball_y != 200):
            vector_ball_y[ball_y] = 1

        vector_ball_directed_velocity_x = cp.zeros(50)
        vector_ball_directed_velocity_x[ball_directed_velocity_x+24] = 1
        
        vector_ball_directed_velocity_y = cp.zeros(50)
        vector_ball_directed_velocity_y[ball_directed_velocity_y+24] = 1

        encoded = cp.concatenate((vector_paddle_pos,vector_ball_x,vector_ball_y,vector_ball_directed_velocity_x,vector_ball_directed_velocity_y))

        return encoded

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
    def test(self, episodes, render = None, model_name = "three_layers", optimizer = ut.Adam, initializator = ut.xavier_initialization, activation = ly.Relu, hidden_layer_size = 16):
        # Create Pong instance
        env = gym.make('PongNoFrameskip-v4',
                        render_mode=render, 
                        obs_type="grayscale")
        
        # Initializing constants
        num_states = 5 #580 #5 # size of the preprocessed input
        num_actions = 2 # up, down and stay

        
        # Initialize Neural Network
        policy_dqn = create_network(in_states=num_states, h1_nodes=hidden_layer_size, out_actions=num_actions, batch_size = self.mini_batch_size, model_name=model_name, weight_initializor = initializator, optimizer = optimizer, activation_function = activation)        

        # Print model architecture
        policy_dqn.print_name()
        
        # Loading the model
        with open("pong_experimental_dql.pkl", 'rb') as file:
            policy_model = pickle.load(file)
        policy_dqn.set_weights(policy_model)

        rewards_test = []
        # Testing
        for _ in tqdm.tqdm(range(episodes)):
            state = env.reset()[0]  # Initialize to state 0
            terminated = False      
            truncated = False  
            rewards = 0

            # Track number of steps taken. Used for terminating early
            steps_truncation = 1 

            previous_state = deque(maxlen=self.size_of_velocity_memory)
            
            for _ in range(4):
                # Until previous state memory is full, we stand still
                action =  0
            
                # Execute action
                new_state,reward,_,_,_ = env.step(action+2) # We map the actions 0-2 to 1-3
            
                rewards += reward

                # Fill previous state memory
                previous_state.append(state)
            
                # Move to the next state
                state = new_state
            
            while (len(cp.nonzero(state[35:195] == 236)[0]) == 0):
                # Until the ball appears, we stand still
                action =  0
            
                # Execute action
                new_state,reward,_,_,_ = env.step(action+2) # We map the actions 0-2 to 1-3
            
                # Update previous state memory
                previous_state.append(state)

                rewards += reward

                # Move to the next state
                state = new_state
                print("No ball")
            

            # Agent navigates map until it falls into a hole (terminated), reaches goal (terminated), or has taken 200 actions (truncated).
            while(not terminated and not truncated):  

                # Preprocess state
                preprocessed_frame = self.state_to_dqn_input(state,previous_state[0])

                # Debug log
                # if (random.random()<0.05):
                #     print("State: ", preprocessed_frame)

                # Select best action   
                action = policy_dqn.forward(preprocessed_frame).argmax().item()
                
                # Debug log
                # print("Chosen action: ", action)

                # Execute action
                state,reward,terminated,truncated,_ = env.step(action+2) # We map the actions 0-2 to 1-3

                if(steps_truncation == -10):
                    truncated = True 
                    steps_truncation = 1

                rewards += reward

                # Update previous state memory
                previous_state.append(state)
                
                # Increment step counter
                steps_truncation += 1

            rewards_test.append(rewards)
  
        print(rewards_test)
        

        # Closing the environment
        env.close()

    
if __name__ == '__main__':
    # Initializing
    doTrain = True # Set to 'True' to train a new model
    doTest = True # Set to 'True' to load and test newest model
    
    
    render_training = None # set to 'human' to see the training on a gaming screen
    render_testing = None # set to 'human' to see the testing on a gaming screen
    test_run_number = 10 # How often we let the agent show what it learned

    # Choice of model
    models = ["two_layers", "three_layers"] 
    model_name = models[1]

    # Choice of optimizer
    optimizers = [ut.Adam, ut.PrimitiveOptimizer] 
    optimizer_name = optimizers[0]

    # Choice of weight initialization
    weight_initializations = [ut.xavier_initialization, ut.old_xavier_initialization, ut.xavier_initialization_uniform, ut.xavier_initialization_normal, ut.simple_initialization, ut.kaiming_initialization] 
    initializator_name = weight_initializations[-1]

    # Choice of activation function
    activation_functions = [ly.Relu, ly.LeakyRelu, ly.Elu, ly.Selu, ly.Sigmoid,  ly.Sigmoid2, ly.Swish, ly.Tanh, ly.Atanh, ly.Sinusoid, ly.Cosinusoid, ly.Gaussian, ly.Softplus, ly.Identity, ly.Prelu]
    activation_name = activation_functions[-1]

    hidden_layer_size = 100 # default: 200
    epoch_number = 4_000 # default: 

    total_start = time.time()



    # Initialize training class
    pong = PongDQL()
    
    # Training 
    if(doTrain == True):
        pong.train(
            epoch_number, 
            render = render_training, 
            model_name = model_name,
            initializator=initializator_name,
            optimizer = optimizer_name,
            hidden_layer_size = hidden_layer_size,
            activation = activation_name
            )
    
    # Testing 
    if(doTest == True):
        pong.test(test_run_number,
                    render = render_testing, 
                    model_name = model_name,
                    initializator=initializator_name,
                    optimizer = optimizer_name,
                    hidden_layer_size= hidden_layer_size,
                    activation = activation_name) 

    
    
    print(" ")


    # Measuring time
    total_end = time.time()
    print("Training and testing took: ", total_end - total_start)


    
