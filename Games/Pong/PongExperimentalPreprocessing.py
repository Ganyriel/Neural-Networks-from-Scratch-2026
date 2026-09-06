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
    # Hyperparameters
    discount_factor_g = 0.9         # discount rate of reward (gamma), default: 0.9  
    network_sync_rate = 40_000          # number of steps the agent takes before syncing the policy and target network, default: 
    replay_memory_size = 20_000       # size of replay memory, default:
    mini_batch_size = 128        # size of the training data set sampled from the replay memory, default: 32
    size_of_velocity_memory = 6 # for calculating the velocity and direction of the ball

    # Train the Pong environment
    def train(self, episodes, render = None, model_name = "three_layers",  optimizer = ut.Adam, initializator = ut.xavier_initialization, activation = ly.Relu, hidden_layer_size = 16):
        
        # Creating environment
        env = gym.make('PongNoFrameskip-v4',
                        render_mode=render, 
                        obs_type="grayscale")
        
    
        # Initializing constants
        num_states = 580 #580 # 5 # size of the preprocessed input (position of paddle, position of ball x2, velocity of ball x2)
        num_actions = 2 # size of action space (up and down)

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

        # List to keep track of points scored or lost. Initialize list to 0's.
        point_scored_per_episode = cp.zeros(episodes)

        # List to keep track of epsilon decay
        epsilon_history = []

        # List to monitor time of an episode
        monitor_time = []

        # List to keep track of loss
        loss_list = []   

        # Track number of steps taken. Used for syncing policy => target network.
        network_step_count=0

        # Keep track of best reward
        best_rewards = -100

        for i in tqdm.tqdm(range(episodes)):
            episode_time = time.time()
            state = env.reset()[0]  # Initialize to state 0
            terminated = False      # True when game ends
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

            # Preprocess state
            not_encoded_new_state = self.state_to_data(new_state, previous_state[0])
            preprocessed_new_state = self.encode_state(*not_encoded_new_state)

            # Agent plays until the game ends or they have taken a certain amount of actions (truncated).
            while(not terminated and not truncated):

                # Keep track of scored points
                point_scored = 0


                # Preprocess state
                not_encoded_state = not_encoded_new_state
                preprocessed_state = preprocessed_new_state

                # Select action based on epsilon-greedy
                if random.random() < epsilon:
                    # Exploration
                    action =  random.randint(0, 1) 
                else:
                    # Exploitation
                    action = policy_dqn.forward(preprocessed_state).argmax().item()
                    
                # Execute action
                new_state,reward,terminated,truncated,_ = env.step(action+2) # We map the actions 0-1 to 2-3


                # Keep track if point was scored TODO!!
                if(reward != 0):
                    point_scored = reward


                # Calculate custom reward
                reward = self.reward_scheduler(state = state, reward=reward, action=action)



                # Preprocessing new state
                not_encoded_new_state = self.state_to_data(new_state, previous_state[0])
                preprocessed_new_state = self.encode_state(*not_encoded_new_state)

                # Keep track of the rewards collected per episode.
                rewards_per_episode[i] += reward

                # Keep track of the points collected per episode.
                point_scored_per_episode[i] += point_scored


                # Save experience into memory                
                if(not_encoded_state[3] == 0 and not_encoded_state[4] == 0):
                    # If no ball, save only some transitions
                    if(random.random()<0.2):
                        memory.append((preprocessed_state, action, preprocessed_new_state, reward, terminated, point_scored)) 
                elif(not_encoded_state[3] <= 0):
                    # If ball is flying left, save only some transitions
                    if(random.random()<0.2):
                        memory.append((preprocessed_state, action, preprocessed_new_state, reward, terminated, point_scored)) 
                else:
                    memory.append((preprocessed_state, action, preprocessed_new_state, reward, terminated, point_scored)) 

                # Update previous state memory
                previous_state.append(state)

                # Move to the next state
                state = new_state

                # Truncate if too many steps taken
                if(steps_truncation == 1_200): 
                    truncated = True 
                    steps_truncation = 0
                
                # Increment step counter
                network_step_count+=1
                steps_truncation += 1

            # Keep track of highest reward
            if rewards_per_episode[i]>best_rewards:
                best_rewards = rewards_per_episode[i]
 
            # Check if enough experience has been collected
            if (len(memory) > self.mini_batch_size) :
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

            # Calculate time for episode
            monitor_time.append(time.time() - episode_time)    

        # Saving the model
        with open("pong_experimental_dql.pkl", 'wb') as file:
            pickle.dump(policy_dqn.get_weights(), file)

        # Print best reward
        print("Best reward: ", best_rewards)
        
        # Close environment
        env.close()

        
        # Plotting
        # Create new graph 
        fig, ax = plt.subplots(3, 2)
        
        # Plot rewards in every episode
        ax[0, 0].plot(rewards_per_episode)
        ax[0, 0].set_title("Rewards per episode")
        
        # Plot time it for each episode  
        ax[0, 1].scatter(cp.arange(episodes)+1,monitor_time)
        ax[0, 1].set_title("Episode time")
        
        # Plot the loss
        loss_list = cp.array(loss_list)
        loss_list[loss_list > 2] = 2
        ax[1, 0].plot(loss_list)
        ax[1, 0].set_title("Loss per episode")

        # Plot epsilon decay (Y-axis) vs episodes (X-axis)
        ax[1, 1].plot(epsilon_history)
        ax[1, 1].set_title("Epsilon in each episode")


        # Plot epsilon decay (Y-axis) vs episodes (X-axis)
        ax[2, 0].plot(point_scored_per_episode)
        ax[2, 0].set_title("Points per episode")
        
        # Save plots
        fig.tight_layout(h_pad = 2, w_pad  = 2)
        fig.savefig('pong_experimental_dql.png')

        # Prints the points scored of the last 100 games
        print("Average points scored in the last 50 games: ", cp.sum(point_scored_per_episode[-51:-1:])/50)

    # Optimize policy network
    def optimize(self, mini_batch, policy_dqn, target_dqn):
        current_q_list = []
        target_q_list = []
        
        for preprocessed_state, action, preprocessed_new_state, reward, terminated, point_scored in mini_batch:

            if point_scored != 0: #terminated: 
                # When point got scored, target q value should be set to the reward.
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

        # Returning loss
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

    def state_to_data(self,state,prev_state):
        # Extracts all the necessary information from the state
             
        # Positions of paddle and ball
        paddle_pos, ball_pos = self.get_positions(state)
        _, prev_ball_pos = self.get_positions(prev_state)

        if(ball_pos == (200,200) or prev_ball_pos == (200,200)):
            ball_absolute_velocity = 0
            ball_directed_velocity = cp.array([0,0])
        else:
            ball_directed_velocity = cp.array(ball_pos)-cp.array(prev_ball_pos)
            ball_absolute_velocity = cp.linalg.norm(ball_directed_velocity)
            if (ball_absolute_velocity == 0):
                ball_directed_velocity = cp.array([0,0])

        # Return the data
        return paddle_pos, ball_pos[0], ball_pos[1], ball_directed_velocity[0], ball_directed_velocity[1]

    def state_to_dqn_input(self, state, prev_state):
        # Performs preprocessing steps

        preprocessed_state = self.state_to_data(state=state, prev_state=prev_state)
        preprocessed_state = self.encode_state(*preprocessed_state)
        return  preprocessed_state

    def reward_scheduler(self, state, reward, action):
        # Function for setting custom rewards


        # reward for scoring a point
        if(reward == 1):
            reward = 5

        # reward for paddle being close to the ball
        paddle_pos, ball_pos = self.get_positions(state)
        paddle_pos = cp.array([140,paddle_pos])
        ball_pos = cp.array(ball_pos)
        

        
        paddle_ball_distance = cp.linalg.norm(paddle_pos-ball_pos)
        if (paddle_ball_distance == 0):
            paddle_ball_distance = 0.5


        if (paddle_ball_distance <= 20 and ball_pos[0]<140):
            reward += 0.2/(paddle_ball_distance**2)

        

        return reward

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

        


    # Run the Pong environment with the learned policy
    def test(self, episodes, render = None, model_name = "three_layers", optimizer = ut.Adam, initializator = ut.xavier_initialization, activation = ly.Relu, hidden_layer_size = 16):
        # Create Pong instance
        env = gym.make('PongNoFrameskip-v4',
                        render_mode=render, 
                        obs_type="grayscale")
        
        # Initializing constants
        num_states = 580 #580 #5 # size of the preprocessed input
        num_actions = 2 # up and down

        
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
            
            for _ in range(self.size_of_velocity_memory):
                # Until previous state memory is full, we stand still
                action =  0
            
                # Execute action
                new_state,reward,_,_,_ = env.step(action+2) # We map the actions 0-1 to 2-3
            
                rewards += reward

                # Fill previous state memory
                previous_state.append(state)
            
                # Move to the next state
                state = new_state
            

            # Agent plays 
            while(not terminated and not truncated):  

                # Preprocess state
                preprocessed_frame = self.state_to_dqn_input(state,previous_state[0])



                # Select best action   
                action = policy_dqn.forward(preprocessed_frame).argmax().item()
                


                # Execute action
                state,reward,terminated,truncated,_ = env.step(action+2) # We map the actions 0-1 to 2-3

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
    render_testing = 'human' # set to 'human' to see the testing on a gaming screen
    test_run_number = 10 # How often we let the agent show what it learned

    # Choice of model
    models = ["two_layers", "three_layers"] 
    model_name = models[0]

    # Choice of optimizer
    optimizers = [ut.Adam, ut.PrimitiveOptimizer] 
    optimizer_name = optimizers[0]

    # Choice of weight initialization
    weight_initializations = [ut.xavier_initialization, ut.old_xavier_initialization, ut.xavier_initialization_uniform, ut.xavier_initialization_normal, ut.simple_initialization, ut.kaiming_initialization] 
    initializator_name = weight_initializations[0]

    # Choice of activation function
    activation_functions = [ly.Relu, ly.LeakyRelu, ly.Elu, ly.Selu, ly.Sigmoid,  ly.Sigmoid2, ly.Swish, ly.Tanh, ly.Atanh, ly.Sinusoid, ly.Cosinusoid, ly.Gaussian, ly.Softplus, ly.Identity, ly.Prelu]
    activation_name = activation_functions[0]

    hidden_layer_size = 580 #default: 580
    epoch_number = 50 # default: 2_000?


    # Measure time
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


    
