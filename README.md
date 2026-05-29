# Neural-Networks-from-Scratch-2026
Advanced Practical called "Neural Networks from Scratch"


FrozenLakeNeural.py contains a Neural Network which is supposed to learn to play Frozen Lake without any Holes. 
The code is currently:
- a bit messy
- doesn't raise an error
- agent solves the puzzle only sometimes (now with 70% of the times)


Temporary stuff:
- Frozen Lake doesn't have holes (for fixing errors)
- The file uses numpy instead of CuPy (Cupy is significantly slower than numpy. Probably due to copying (gpu mostly copies))


Possible tweakable stuff:
- hidden layer size (3 spots: 2 in training and 1 in testing)
- learning rate
- learning rate change
- rewards
- batch_size
- number of epochs

Possible ToDos:
- implement new activation function (relu instead of sigmoid maybe? Requires change to gradient)
- implement different optimizers


