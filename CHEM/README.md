# Learning By Doing - NeurIPS 2021 Competition CHEM Track Solution


## Instructions

To reproduce my solution for the CHEM track:

1. Clone this repository
2. Download the training data and starter kit files from [codalab](https://competitions.codalab.org/competitions/33378#participate-get_starting_kit) and unpack them in the `CHEM` folder (should create new folders named `CHEM_training_data` and `CHEM_starter_kit`)
3. Run the scripts described below

## Contents

The following 3 interactive scripts whould be run in order. These are similar to jupyter notebooks and were meant to be run in an interactive session (on an IDE that supports it) but it should also be possible to run them in sequence to generate the artifacts necessary:

* **structure-inference.py**: Infers the structure of the chemical network, i.e., what reactions are present. Also provides an initial estimate of the dynamical system parameters;
* **inference.py**: Infers the parameters of the dynamical system (reaction rates, input matrix and initial conditions) given a fixed structure;
* **control.py**: Optimizes the open loop control to provide to each test system.

The scripts make use of the the file **utils.py** which contains the functions to load training and test data and create the submission.

## Requirements

The following libraries should be installed in order to run the scripts (the versions specified are simply the ones I used):

* python (3.7.10)
* numpy (1.20.3)
* scipy (1.7.1)
* pandas (1.2.5)
* jax (0.2.17)
* jaxlib (0.1.68)
* matplotlib (3.3.4)
* seaborn (0.11.1)