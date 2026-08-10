# Neural-Networks-from-Scratch-2026
Advanced Practical called "Neural Networks from Scratch"

## Files to be sorted
#### BlackJackQLearning
contains an hands-on approach to Q-Learning and simpler implementation of RL.
This approach was chosen to get wider understanding of different possibilites on how to move forward.
This file was created using the Tutorial on the [Gymnasium Library Documentation ](https://gymnasium.farama.org/)


## Easier-Games Folder
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

#### FrozenLakeNeuralCupy
contains old implementation of a DeepQ approach to train a Neural Network to solve FrozenLake in CuPy.

Issue: Cupy is significantly slower than numpy. Probably due to copying (gpu mostly copies)



### BlackJack Folder 
#### BlackJackDeepQ
contains an unclean implementation of deep Q learning for Blackjack. The agent's success rate is 40-42%. Requires some cleanup and hyperparameter tuning



### MountainCar Folder 
#### MountainCarDeepQ
contains a training file for MountainCar Deep Q learning. It has very high success rate.



## Atari-Games Folder
### Pong Folder
#### PongDeepQ
contains an incomplete implementation of Deep Q learning for Pong. It is currently not working (testing often has no ball for unknown reason)



## Durak Folder
#### DurakNotes.odt
contains a description of the game and design ideas for a agent and environment.

#### DurakQLearning
contains a sample for creating a Durak Q Learning approach.
Has to be altered and adjusted to fit the Environment. We are gonna see how that plays out.

#### DurakEnvironment
contains a not yet fully implemented Environment for Durak. Yet it serves as a sample, that still needs a lot adjjustme>
I'm doing this to understand custom environment creation and wrappers etc.
After some research and game analysis and so on. Durak seems not that easy to solve.


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
