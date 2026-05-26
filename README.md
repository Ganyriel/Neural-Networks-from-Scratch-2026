# Neural-Networks-from-Scratch-2026
Advanced Practical called "Neural Networks from Scratch"


FrozenLakeNeural.py contains a Neural Network which is supposed to learn to play Frozen Lake without any Holes. 
The code is currently:
- messy
- doesn't raise an error
- raises warnings
- doesn't work (i.e. the agent only learned once)
- has broken Reward stuff (plot shows the agent is getting a reward of no more than 1, though it should get +0.01 for each step alone)

Temporary stuff:
- rewards are set wrong (for fixing the errors)
- Frozen Lake doesn't have holes (for fixing errors)
- The file uses numpy instead of CuPy (will be changed after bugs are fixed)
