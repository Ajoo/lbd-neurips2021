#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Designs the controllers for the beetle robots.

@author: ajoo
"""
import pandas as pd
import numpy as np

from functools import partial
from collections import defaultdict
import pickle

import jax.numpy as jnp
from jax import jit, vmap

from utils import load_train
from dynamics import to_continuous, to_polar, rolling_mean, beetle_dynamics, g
from control import beetle_pd_controller as pd_controller
from testbench import euler, simulate

#%% Load System Identification Results

SUBMISSION_FOLDER = 'submissions/submission'
SYSID_RESULTS = 'beetle-results.pickle'

beetle_variations = ['great', 'rebel']
input_variations = ['devious', 'mauve', 'vivacious', 'wine']

h = 0.01

with open(SYSID_RESULTS, 'rb') as file:
    results = pickle.load(file)

#%% Save Input Matrices
for beetle in beetle_variations:
    for input_var in input_variations:

        beetle_params = results[beetle][input_var]
        Bs = beetle_params['Bs']
        Br = beetle_params['Br']
        
        Kg = np.linalg.inv(Bs)

        np.savetxt(
            '{}/{}-{}-beetle.txt'.format(SUBMISSION_FOLDER, beetle, input_var),
            np.concatenate((Kg, Br), axis=1)
            )

#%% Design Controller

# objective scalarization coeficient
CONTROL_COEF = 0.5

# kp and kd grid
kp = np.logspace(1, 5, 30)
kd = np.logspace(1, 3, 30)

pd_params = {}
objective = {}
param_names = ['l1', 'l2', 'j1', 'j2', 'mu1', 'd1', 'B']
for beetle in beetle_variations:
    for input_var in input_variations:
        print('Designing controller for:', beetle, input_var)

        # Load training data
        _, x, v, _ = load_train('{}-{}'.format(beetle, input_var), 'beetle')
        
        # get initial state for each trajectory (in joint space)
        tt1, w1 = to_polar(x[[0],1], v[[0],1])
        tt2, w2 = to_polar(x[[0],0] - x[[0],1], v[[0],0] - v[[0],1])
        initial_state = np.concatenate((tt1, tt2, w1, w2), axis=0)
        # use the trajectory x2 as reference
        ref = x[:,0]         
                
        # get system identification parameters
        beetle_params = results[beetle][input_var]
        params = [beetle_params[k] for k in param_names]
        l1, l2, j1, j2, mu1, d, B = params
        Bs = beetle_params['Bs']
        Br = beetle_params['Br']
        Kg = np.linalg.inv(Bs)

        # get dynamics for this beetle
        system_dynamics = partial(beetle_dynamics, params)
        
        # define metrics function
        def metrics(tunable_controller_params, initial_state, ref):
            Kp, Kd = tunable_controller_params
            # collect all params necessary for the controller 
            controller_params = (l1, l2, mu1, Kg, Kp, Kd, Br.T)
            controller_func = partial(pd_controller, controller_params)
            
            # create closed loop system simulator
            # (using 10 euler steps per controller step)
            simulator = simulate(system_dynamics, controller_func, h, 10)
            # simulate closed loop system
            _, (ctrl_input, xe, _) = simulator(initial_state, ref, jnp.array([np.nan, np.nan]))
            # compute tracking error and input energy
            tracking_error = xe[1:] - ref[:-1]
            tracking_error = h*np.sum(tracking_error**2, axis=(0,1))
            input_energy = h*np.sum(ctrl_input[:-1]**2, axis=(0,1))
            return tracking_error, input_energy

        # vectorize metrics function
        vmetrics = jit(vmap(metrics, (None, -1, -1), (-1, -1))) # over all instances
        vmetrics = jit(vmap(vmetrics, ((0, None), None, None), (0, 0))) # over Kp values -> axis 1
        vmetrics = jit(vmap(vmetrics, ((None, 0), None, None), (0, 0))) # over Kd values -> axis 0
        
        # compute metrics for all instances and controller param values
        deviation, input_energy = vmetrics((kp, kd), initial_state, ref)

        # mean over instances
        mean_deviation = np.mean(deviation, -1)
        mean_input_energy = np.mean(input_energy, -1)
        
        # scalarized objective
        obj = np.array(CONTROL_COEF*mean_input_energy + mean_deviation)
        obj[np.isnan(obj)] = np.inf
        
        # find minimizing kp and kd indexes
        kd_idx = np.argmin(obj, axis=0)
        obj = np.take_along_axis(obj, kd_idx[None], 0).flatten()
        kp_idx = np.argmin(obj)
        kd_idx = kd_idx[kp_idx]
        
        # save parameters and objectives
        pd_params['{}-{}'.format(beetle, input_var)] = (kp[kp_idx], kd[kd_idx])
        objective['{}-{}'.format(beetle, input_var)] = (mean_deviation[kd_idx, kp_idx], mean_input_energy[kd_idx, kp_idx])
        
        print('gains:', kp[kp_idx], kd[kd_idx])
        print('loss:', mean_deviation[kd_idx, kp_idx], mean_input_energy[kd_idx, kp_idx])
        
#%% Save Results
import json

with open('{}/beetle-pd.txt'.format(SUBMISSION_FOLDER), 'w') as file:
    json.dump(pd_params, file)
