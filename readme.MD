#Pytorch RKN Implementation 

## Setup
Python 3.6.

## Code overview:
This code is structured such that you can use the the actual RKN cell on its own for maximum flexibility. For an example 
how to use it see experiments/balls/cell_usage_tutorial.py

The RKN, as described in the paper is extra in the corresponding directories.

### rkn
Implementation of the RKN, as described in the 2019 ICML Paper 
- I also changed the evaluation metric from rmse to mse as this is a bit cleaner when averaging over batches, if you 
want the rmse, just compute it yourself from that.
###Other Parts of the Code

### util
Utility functionality not specific to the RKN.

### experiments
Currently Implemented Experiments
- Balls: Ball Tracking, not published in any paper yet. There are currently two versions of this:
  1. cell_usage_tutorial: Showcases how to use the RKN_cell as a standalone module (here to rebuild the RKN)
  2. pendulum_balls.py: Uses the full RKN implementation provided in "rkn"

### tests
Some simple tests as sanatiy checks. 

 