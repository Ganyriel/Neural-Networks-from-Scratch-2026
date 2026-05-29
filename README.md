# Neural-Networks-from-Scratch-2026
Advanced Practical called "Neural Networks from Scratch"


FrozenLakeNeural.py contains a Neural Network which is supposed to learn to play Frozen Lake without any Holes. 
The code is currently:
- a bit messy
- agent solves the puzzle only sometimes (now (29.05) with 70% of the times)
- MSE Loss has probably an error in the gradient

Temporary stuff:
- Frozen Lake doesn't have holes (for fixing errors)
- The file uses numpy instead of CuPy (Cupy is significantly slower than numpy. Probably due to copying (gpu mostly copies))


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


