#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Designs the controllers for the butterfly robots.

@author: ajoo
"""
import pandas as pd
import numpy as np

from functools import partial
from collections import defaultdict
import pickle

import jax.numpy as jnp
from jax import jit, vmap

from utils import load_instance, load_train
from dynamics import to_continuous, to_polar, rolling_mean, relative_to, butterfly_dynamics, g
from control import butterfly_pd_controller as pd_controller
from testbench import euler, simulate

#%% Load System Identification Results

SUBMISSION_FOLDER = 'submissions/submission'
SYSID_RESULTS = 'butterfly-results.pickle'

butterfly_variations = ['talented', 'thoughtful']
input_variations = ['antique', 'ruddy', 'steel', 'zippy']

h = 0.01

with open(SYSID_RESULTS,'rb') as file:
    results = pickle.load(file)

#%% Save Input Matrices

for butterfly in butterfly_variations:
    for input_var in input_variations:

        butterfly_params = results[butterfly][input_var]
        Bs = butterfly_params['Bs']
        Br = butterfly_params['Br']
        
        Kg = np.linalg.inv(Bs)
        
        np.savetxt(
            '{}/{}-{}-butterfly.txt'.format(SUBMISSION_FOLDER, butterfly, input_var),
            np.concatenate((Kg, Br), axis=1)
            )

#%% Design Controller

# objective scalarization coeficient
CONTROL_COEF = 1.0

# kp and kd grid
kp = np.logspace(1, 5, 30)
kd = np.logspace(0, 2, 30)

pd_params = {}
objective = {}
for butterfly in butterfly_variations:
    km = 0.0 if butterfly=='thoughtful' else 10.0 # maneuverability "gain"
    
    for input_var in input_variations:
        print('Designing controller for:', butterfly, input_var)

        # Load training data
        _, x, v, _ = load_train('{}-{}'.format(butterfly, input_var), 'butterfly')
        
        # get initial state for each trajectory (in joint space)
        tt1, w1 = to_polar(x[[0],2], v[[0],2])
        tt2, w2 = to_polar(x[[0],1] - x[[0],2], v[[0],1] - v[[0],2])
        tt3, w3 = to_polar(x[[0],0] - x[[0],1], v[[0],0] - v[[0],1])
        initial_state = np.concatenate((tt1, tt2, tt3, w1, w2, w3), axis=0)
        # use the trajectory x3 as reference
        ref = x[:,0]
        
        # get system identification parameters
        butterfly_params = results[butterfly][input_var]
        l1, l2, l3 = butterfly_params['l1'], butterfly_params['l2'], butterfly_params['l3']
        m = butterfly_params['j']
        Mt = np.diag(m)
        j12, j13, j23 = butterfly_params['j12'], butterfly_params['j13'], butterfly_params['j23']
        Mt += np.array([[0.0, j12, j13], [j12, 0.0, j13], [j13, j13, 0.0]])
        d = np.mean(butterfly_params['d'])
        B = butterfly_params['B']
        Bs = butterfly_params['Bs']
        Br = butterfly_params['Br']
        Kg = np.linalg.inv(Bs)

        # get dynamics and controller for this butterfly
        system_dynamics = partial(butterfly_dynamics, (l1, l2, l3, Mt, d, B))
        controller = partial(pd_controller, (l1, l2, l3, j12, j13, Kg, Br))
        
        # define metrics function
        def metrics(controller_params, initial_state, ref):
            controller_func = partial(controller, controller_params)
            
            # create closed loop system simulator
            # (using 10 euler steps per controller step)
            simulator = simulate(system_dynamics, controller_func, h, 10)
            # simulate closed loop system
            _, (ctrl_input, xe, _) = simulator(initial_state, ref, None)
            # compute tracking error and input energy
            tracking_error = xe[1:] - ref[:-1]
            tracking_error = h*np.sum(tracking_error**2, axis=(0,1))
            input_energy = h*np.sum(ctrl_input[:-1]**2, axis=(0,1))
            return tracking_error, input_energy
        
        # vectorize metrics function
        vmetrics = jit(vmap(metrics, (None, -1, -1), (-1, -1))) # over all instances
        vmetrics = jit(vmap(vmetrics, ((0, None, None), None, None), (0, 0))) # over Kp -> axis 1
        vmetrics = jit(vmap(vmetrics, ((None, 0, None), None, None), (0, 0))) # over Kd -> axis 0
        
        # compute metrics for all instances and controller param values
        deviation, input_energy = vmetrics((kp, kd, km), initial_state, ref)

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
        pd_params['{}-{}'.format(butterfly, input_var)] = (kp[kp_idx], kd[kd_idx], km)
        objective['{}-{}'.format(butterfly, input_var)] = (mean_deviation[kd_idx, kp_idx], mean_input_energy[kd_idx, kp_idx])
        
        print('gains:', kp[kp_idx], kd[kd_idx], km)
        print('loss:', mean_deviation[kd_idx, kp_idx], mean_input_energy[kd_idx, kp_idx])
#%% Save Results
import json

with open('{}/butterfly-pd.txt'.format(SUBMISSION_FOLDER), 'w') as file:
    json.dump(pd_params, file)