# Learning By Doing - NeurIPS 2021 Competition ROBO Track Solution


## Instructions

To reproduce the solutions for the ROBO track:

1. Clone this repository
2. Download the training trajectories file from [codalab](https://competitions.codalab.org/competitions/33622#participate) and unpack them in the `ROBO` folder (should create a new folder named `training_trajectories`)
3. Run the scripts that generate the artifacts required for the final submission (in the `ROBO/submissions/submission` folder) described in the next section
4. Zip the `ROBO/submissions/submission` folder.

A short presentation describing the solution can be found [here](beamer.pdf).

## Contents

For each robot type (bumblebee, beetle and butterfly) there are the following scripts:

* **[robot type]-model.py**: Performs system identification for all instances of the robot type (i.e., estimated the parameters of the dynamical system model);
* **[robot type]-control.py**: Designs the controller for all instances of the robot type.

Running these 2 scripts in sequence should produce the necessary artifacts to reproduce the final submission.

The scripts make use of additional utility files developed for this competition:

* **utils.py**: contains functions to load the training data;
* **dynamics.py**: dynamic models used to simulate the robots along with some helper functions for system identification;
* **control.py**: controllers and functions for controller design;
* **testbench.py**: functions to simulate both the open and closed loop system plus utilities to vizualize the simulations.

## Requirements

The following libraries should be installed in order to run the scripts (the versions specified are simply the ones I used):

* python (3.7.10)
* numpy (1.20.3)
* scipy (1.7.1)
* pandas (1.2.5)
* jax (0.2.17)
* jaxlib (0.1.68)

Optional:

* matplotlib (3.3.4)
* seaborn (0.11.1)
* imageio (2.9.0)