# Neural-Networks-from-Scratch-2026
Advanced Practical called "Neural Networks from Scratch"

## Games Folder

### Shared_files Folder
#### layers_numpy
contains implementation of neural network layers

#### utils_numpy
contains helper functions as well as optimizers and weight initializations

#### networks_numpy
contains functions for creating neural networks

#### run_logger
contains a class for functions to log during train


### FrozenLake Folder
#### FrozenLakeDeepQ
contains a Neural Network which is supposed to learn to play Frozen Lake using Deep Q Learning. 

The code currently:
- is slightly messy
- runs 1_000 epochs in a few seconds
- slows down considerably when running many (few hundred) experiments (1_000 experiments should have taken ~2_000s, but took 3325s)

Temporary stuff:
- The file uses numpy instead of CuPy (Cupy is significantly slower than numpy. Probably due to copying (gpu mostly copies))

Remark:
- agent solves the puzzle almost always (19.06 with 99.96% chance, tested over 300 experiments)



### BlackJack Folder 
#### BlackJackDeepQ
contains an implementation of deep Q learning for Blackjack. The agent's success rate is 40-42%. Requires some cleanup and hyperparameter tuning

#### BlackJackQLearning
contains an hands-on approach to Q-Learning and simpler implementation of RL.
This approach was chosen to get wider understanding of different possibilites on how to move forward.
This file was created using the Tutorial on the [Gymnasium Library Documentation ](https://gymnasium.farama.org/)



### MountainCar Folder 
#### MountainCarDeepQ
contains a training file for MountainCar Deep Q learning. It has very high success rate. 
Todo: weight initialization and optimizers


### Pong Folder

#### Folders
The cupy and numpy folders contain identical files, but ones are implemented in numpy and the others in cupy

#### PongNumpy
contains an implementation of Deep Q learning for Pong using Numpy. 

Issues:
- the learning takes too long for good testing
- it is uncertain whether the training works as intended, since the reset of the environment seems off after losing a point (it resets to the same state)
- is incompatible with convolutional neural networks ("convolutional" and "custom")
- Todo: weight initialization and optimizers



#### PongDeepFrameSkipNumpy
contains an incomplete implementation of Deep Q learning for Pong with SkipFrames from https://danieltakeshi.github.io/2016/11/25/frame-skipping-and-preprocessing-for-deep-q-networks-on-atari-2600-games/

It has similar issues as PongDeepQNumpy, but is compatible with convolutional neural networks

#### PongExperimentalPreprocessing
contains an almost finished implementation of Deep Q learning for Pong. It uses very strong preprocessing.
It requires hyperparameter tuning.


## Ideas for improving training

### Possible tweakable stuff:
- hidden layer size
- learning rate
- learning rate change
- rewards
- batch_size
- number of epochs
- synchronization frequency

### Possible ToDos:
- implement new activation functions
- implement different optimizers
- implement different weight initialization methods
