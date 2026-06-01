# Neural-Networks-from-Scratch-2026
Advanced Practical called "Neural Networks from Scratch"

## Files with \<name>.py
#### FrozenLakeNeural
contains a Neural Network which is supposed to learn to play Frozen Lake without any Holes. 
The code is currently:
- a bit messy
- agent solves the puzzle only sometimes (now (29.05) with 70% of the times)
- MSE Loss has probably an error in the gradient

Temporary stuff:
- Frozen Lake doesn't have holes (for fixing errors)
- The file uses numpy instead of CuPy   
    - CuPy is significantly slower than numpy. The GPU's parallelism is underutilized due to the tiny problem size. 
    Copying could be the issue, because the CPU sending the GPU a command has a fixed cost.


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