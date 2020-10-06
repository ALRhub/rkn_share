# Pytorch RKN Implementation 

## Setup
Python 3.6, Requirements, see req.txt

Before any experiments can be executed, you need to generate the data. In corresponding experiment folder there is a data folder with code to do that. 
To generate the data for the quad_link experiments, you need the "n_link_sim" simulator:
https://github.com/LCAS/RKN/tree/master/n_link_sim

## Code overview:
This code is structured such that you can use the actual RKN cell (i.e. the recurrent part) on its own for maximum flexibility. For an example 
how to use it see experiments/balls/cell_usage_tutorial.py

The RKN, as described in the paper is extra in the corresponding directories.

### rkn
Implementation of the RKN, as described in the 2019 ICML Paper 
- I changed the evaluation metric from rmse to mse as this is a bit cleaner when averaging over batches, if you 
want the rmse, just compute it yourself from that.

### Other Parts of the Code

### util
Utility functionality not specific to the RKN.

### experiments
Currently Implemented Experiments
- Balls: Ball Tracking, not published in any paper. There are currently two versions of this:
  1. cell_usage_tutorial: Showcases how to use the RKN_cell as a standalone module (here to rebuild the RKN)
  2. pendulum_balls.py: Uses the full RKN implementation provided in "rkn"
  
- Pendulum and QuadLink state estimation from the Paper.

### tests
Some simple tests as sanatiy checks. 

 
