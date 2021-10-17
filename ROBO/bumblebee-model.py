#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Estimates the bumblebee model parameters.

The results are saved in a pickle file.

@author: ajoo
"""
import pandas as pd
import numpy as np

from utils import load_train
from dynamics import rolling_mean, remove_nullspace
from numpy.linalg import lstsq, svd

#%%
OUTPUT_FILE = 'bumblebee-results.pickle'

bumblebees = ['great-bipedal', 'great-impartial', 'great-piquant', 'great-proficient',
              'lush-bipedal', 'lush-impartial', 'lush-piquant', 'lush-proficient']

h = 0.01
g = np.array([0, -9.81])

results = {}
for bumblebee in bumblebees:
    
    # Load all training trajectories for this variant
    t, x, v, u = load_train(bumblebee, 'bumblebee')
    
    # Get generalized coordinates and velocities from joint positions
    x = np.stack((x[:,0,0], x[:,1,1]), 1)
    v = np.stack((v[:,0,0], v[:,1,1]), 1)
    # Compute generalized acceleration by finite differences
    a = np.diff(v, axis=0)/h - g[:,None]
    # Ignore last input
    u = u[:-1]
    # Apply a 2 sample moving average to the velocities (to estimate velocity at midpoint)
    v = rolling_mean(v)
    
    # Ignore input subspace that is unexplored in training trajectories
    u, Vt, _ = remove_nullspace(u, 2)
    
    # regress accelerations from inputs stacking all trajectories
    # model: a = B@u + eps
    # I also tried adding a damping term but it looks like this robot has none
    Y = np.swapaxes(a, 1, -1).reshape((-1,2))
    X = np.swapaxes(u, 1, -1).reshape((-1,2))
    coefs, r, sigma, _ = lstsq(X, Y)
    B = coefs.T
    
    # Do a "polar" decomposition of B into a symmetric and an orthogonal part
    # B = Bs@Br
    # Right multiply by the projection matrix onto the range of u
    U, s, Vt2 = np.linalg.svd(B)
    Bs = (U*s)@U.T
    Br = U@Vt2@Vt
    B = B@Vt
    
    results[bumblebee] = {'B': B, 'Bs': Bs, 'Br': Br}
    
#%% Save results
import pickle

with open(OUTPUT_FILE, 'wb') as file:
    pickle.dump(results, file)
