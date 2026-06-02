# Neural-Networks-from-Scratch-2026
Advanced Practical called "Neural Networks from Scratch"

## Files with \<name>.py
#### FrozenLakeNeuralAdam
contains a Neural Network which is supposed to learn to play Frozen Lake. 

The code is currently:
- a bit messy
- agent solves the puzzle only almost always (now (02.06) with 84.63% of the times, tested over 1_000 experiments)
- runs 1_000 epochs in a few seconds
- slows down considerably when running many (few hundred) experiments (1_000 experiments should have taken ~2_000s, but took 3325s)

Temporary stuff:
- The file uses numpy instead of CuPy (Cupy is significantly slower than numpy. Probably due to copying (gpu mostly copies))
- unresolved issue with handling learning rate (ADAM uses a static learning rate)

Possible tweakable stuff:
- hidden layer size
- learning rate
- learning rate change
- rewards
- batch_size
- number of epochs
- synchronization frequency

Possible ToDos:
- implement new activation functions
- implement different optimizers

#### BlackJackQLearning
contains an hands-on approach to Q-Learning and simpler implementation of RL.
This approach was chosen to get wider understanding of different possibilites on how to move forward.
This file was created using the Tutorial on the [Gymnasium Library Documentation ](https://gymnasium.farama.org/)

#### DurakQLearning
contains a sample for creating a Durak 

#### DurakEnvironment
contains a not yet fully implemented Environment for Durak. Yet it serves as a sample, that still needs a lot adjjustment.
I'm doing this to understand custom environment creation and wrappers etc.
