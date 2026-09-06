import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random
import pickle
import time
import tqdm
import sys
import os

# get directory and shared_files
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SHARED_DIR = os.path.join(SCRIPT_DIR, '..', 'shared_files')

# make sure path exists
SHARED_DIR = os.path.abspath(SHARED_DIR)


if SHARED_DIR not in sys.path:
    sys.path.insert(0, SHARED_DIR)  # insert path into sys.path with high priority

# file imports
import utils_numpy as ut 
import layers_numpy as ly
from networks_numpy import create_network
from run_logger import RunLogger


# FrozenLake Deep Q-Learning
class FrozenLakeDQL():
    # Hyperparameters (adjustable)
    discount_factor_g = 0.9         # discount rate (gamma),                                                        default: 0.9  
    network_sync_rate = 10          # number of steps the agent takes before syncing the policy and target network, default: 10
    replay_memory_size = 1_000      # size of replay memory,                                                        default: 1_000
    mini_batch_size = 32            # size of the training data set sampled from the replay memory,                 default: 32
    eval_interval = 10              # evaluation intervals for greedy evaluation for example,                       default: 10
    
    # Initializing number of states
    num_states = 0


    # Train the FrozenLake environment
    def train(self, episodes, render = None, is_slippery = False, model_name = "three_layers", optimizer = ut.Adam, initializator = ut.xavier_initialization, activation = ly.Relu, hidden_layer_size = 16):
        
        # initialize the current config for logging
        config = {
            "model_name": model_name,
            "optimizer": optimizer.__name__ if hasattr(optimizer, "__name__") else str(optimizer),
            "initializator": initializator.__name__ if hasattr(initializator, "__name__") else str(initializator),
            "activation function": activation.__name__ if hasattr(activation, "__name__") else str(activation),
            "hidden_layer_size": hidden_layer_size,
            "is_slippery": is_slippery,
            "seed": getattr(self, "seed", 0),
            "output_dir": getattr(self, "output_dir", "results"),
            "episodes": episodes,
        }
        
        logger = RunLogger(config, output_dir=config.get("output_dir", "results"))
        
        # Creating environment
        env = gym.make(
            'FrozenLake-v1', 
            map_name="4x4",
            is_slippery=is_slippery, 
            render_mode=render,
            reward_schedule=(1, 0.0, 0.0)  #default: 1, 0.0, 0.0
        )
        
        env.reset(seed=getattr(self, "seed", 0))         
        env.action_space.seed(getattr(self, "seed", 0)) 
        
        loss_list = []   

        # Initializing constants
        num_states  = env.observation_space.n
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

            
            training_happened = False
            
            # Check if enough experience has been collected and if at least 1 reward has been collected
            if (len(memory) > self.mini_batch_size and np.max(rewards_per_episode) > 0): 
                mini_batch = memory.sample(self.mini_batch_size)
                loss = self.optimize(mini_batch, policy_dqn, target_dqn)
                loss_list.append(loss)
                training_happened = True        
   

                # Decay epsilon
                epsilon = max(epsilon - 1 / episodes, 0.00) # possible augmentation: set a minimum epsilon (i.e. change to 0.1)
                epsilon_history.append(epsilon)

                # Copy policy network to target network after a certain number of steps
                if step_count > self.network_sync_rate:
                    target_dqn.set_weights(policy_dqn.get_weights())
                    step_count=0
                    
            # logging after optimize, so the loss belongs to this episode
            current_loss = loss_list[-1] if training_happened else float("nan")
            logger.log_episode(float(rewards_per_episode[i]), float(current_loss))
            
            # greedy evaluation
            eval_interval = getattr(self, "eval_interval", None)
            if eval_interval and i % eval_interval == 0:
                eval_env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=is_slippery)
                eval_reward = self.evaluate_greedy(policy_dqn, eval_env, num_eval_episodes=10)
                logger.log_eval(i, eval_reward)
                eval_env.close() 
            
        # Close environment
        env.close()
        
        logger.save()
        print(f"[train] Training completed. Results saved to {config['output_dir']}")

        # Saving the model
        with open('policy_dqn.pkl', 'wb') as file:
            pickle.dump(policy_dqn.get_weights(), file)
        

        # Create new graph for plotting only this run. 
        plt.figure(1)
        
        # Plot rewards in every episode
        plt.subplot(221) 
        plt.plot(rewards_per_episode)
        plt.title("Rewards per episode")

        
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
        plt.savefig('frozen_lake_dql.png')

    # Optimize policy network
    def optimize(self, mini_batch, policy_dqn, target_dqn):

        # Get number of input nodes
        num_states = self.num_states #policy_dqn.in_features

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
        
   
    
    def state_to_dqn_input(self, state:int, num_states:int):
        '''
        Converts an state (int) to a tensor representation.
        For example, the FrozenLake 4x4 map has 4x4=16 states numbered from 0 to 15. 

        Parameters: state=1, num_states=16
        Return: tensor([0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
        '''
        input_tensor = np.zeros(num_states)
        input_tensor[state] = 1
        return input_tensor

    # Run the FrozeLake environment with the learned policy
    def test(self, episodes, is_slippery = False, render = None, model_name = "three_layers",  optimizer = ut.Adam, initializator = ut.xavier_initialization, activation = ly.Relu, hidden_layer_size = 16):
        succesful = 0


        # Create FrozenLake instance
        env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=is_slippery, render_mode=render)

        # Initializing
        num_states = env.observation_space.n
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

        
           
        # For keeping track of successesful training sessions   
        if (reward == 1):
            succesful = 1
            # Debugging Logs:
            # print("The agent reached the goal!")

        # Closing the environment
        env.close()

        # Debugging Logs: printing the q-values of the trained network
        # self.print_dqn(policy_dqn)

        # Debug log for testing PRelu
        # print("alpha: ", policy_dqn.get_weights()[1])

        # Returning whether the agent fulfilled their goal
        return succesful

    
    def print_dqn(self, dqn):
        # Print DQN: state, best action, q values

        # Get number of input nodes
        num_states = self.num_states #dqn.in_features

        # for printing 0,1,2,3 => L(eft),D(own),R(ight),U(p)
        ACTIONS = ['L','D','R','U']     

        # Loop each state and print policy to console
        for s in range(num_states):
            #  Format q values for printing
            q_values = ''
            for q in dqn.forward(self.state_to_dqn_input(s, num_states)).tolist():
                q_values += "{:+.2f}".format(q)+' '  # Concatenate q values, format to 2 decimals
            q_values=q_values.rstrip()              # Remove space at the end

            # Map the best action to L D R U
            best_action = ACTIONS[dqn.forward(self.state_to_dqn_input(s, num_states)).argmax()]

            # Print policy in the format of: state, action, q values
            # The printed layout matches the FrozenLake map.
            print(f'{s:02},{best_action},[{q_values}]', end=' ')         
            print() # Print a new line for every state
    
    def evaluate_greedy(self, policy_dqn, env, num_eval_episodes=10):
        rewards = []
        for _ in range(num_eval_episodes):
            state = env.reset()[0]
            total = 0
            terminated = False
            truncated = False
            while not (terminated or truncated):
                action = policy_dqn.forward(self.state_to_dqn_input(state, env.observation_space.n)).argmax().item()
                state, reward, terminated, truncated, _ = env.step(action)
                total += reward
            rewards.append(total)
        return np.mean(rewards)

if __name__ == '__main__':
    # Variables for the environment
    is_slippery_surface = False # set to True to make the agent sometimes doing the wrong movement command, default: False

    # Variables for training and testing
    testing_only = False # Set to 'True' to only load and test newest model
    render_training = None # set to 'human' to see the training on a gaming screen
    render_testing = None # set to 'human' to see the testing on a gaming screen
    test_run_number = 3 # How often we let it show what it learned 

    # Choice of model
    models = ["one_layer", "two_layers", "three_layers"] 
    model_name = models[1]
    hidden_layer_size = 16 # default: 16

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
    number_of_experiments = 50 # How many NNs we train
    epoch_number = 1_000 # default: 1_000 

    total_start = time.time()


    sum_of_successes = 0
    

    for i in np.arange(number_of_experiments)+1:
        print("")
        print("_________________________________________________________________")
        print("Experiment number: ", i, "/", number_of_experiments)

        # Performance logging:
        # start = time.time()

        # Initialize training class
        frozen_lake = FrozenLakeDQL()
        
        # give each experiment a distinct seed and set the output dir
        frozen_lake.seed = int(i)
        frozen_lake.output_dir = "results"

        # seed the RNGs for reproducibility
        random.seed(frozen_lake.seed)
        np.random.seed(frozen_lake.seed)

        
        # Training 
        if(testing_only == False):
            frozen_lake.train(
                epoch_number, 
                render = render_training, 
                is_slippery=is_slippery_surface, 
                model_name = model_name,
                hidden_layer_size = hidden_layer_size,
                initializator=initializator_name,
                optimizer = optimizer_name,
                activation = activation_name)

        

        # Testing and keeping track of successes
        sum_of_successes += frozen_lake.test(test_run_number,
                                            is_slippery=is_slippery_surface, 
                                            render = render_testing, 
                                            model_name = model_name,
                                            hidden_layer_size= hidden_layer_size,
                                            initializator=initializator_name,
                                            optimizer = optimizer_name,
                                            activation = activation_name)
        


    
    print("Number of successes: ", sum_of_successes)
    print("Proportion of successes: ", sum_of_successes/number_of_experiments)

    # Measuring time
    total_end = time.time()
    print("Training took: ", total_end - total_start)


    
