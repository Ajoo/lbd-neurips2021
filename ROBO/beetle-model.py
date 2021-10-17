#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Estimates the beetle model parameters.

The results are saved in a pickle file.

@author: ajoo
"""
import pandas as pd
import numpy as np

from utils import load_train
from dynamics import to_beetle_joint_space, remove_nullspace
from numpy.linalg import lstsq, svd

from collections import defaultdict

#%%
OUTPUT_FILE = 'beetle-results.pickle'

beetle_variations = ['great', 'rebel']
input_variations = ['devious', 'mauve', 'vivacious', 'wine']

h = 0.01
g = 9.81

results = defaultdict(dict)
for beetle in beetle_variations:
    for input_var in input_variations:
        print('Identifying {}-{}'.format(beetle, input_var))
        
        # Load all training trajectories for this variant
        t, x, v, u = load_train('{}-{}'.format(beetle, input_var), 'beetle')

        # Compute link lengths and
        # joint space angles, angular velocities and angular accelerations
        l1, l2, tt, w, a = to_beetle_joint_space(x, v, h)
        tt1, tt2 = tt[:,0], tt[:,1]
        tt21 = tt2 - tt1
        w1, w2 = w[:,0], w[:,1]
        a1, a2 = a[:,0], a[:,1]   
        
        # Ignore input subspace that is unexplored in training trajectories
        u = u[:-1]
        u, Vt, _ = remove_nullspace(u, 2)
        u = np.swapaxes(u, 1, -1)
        
        # Estimate parameters of 1st joint equations of motion
        # model is the following 
        # c1: coriollis + centrifugal term
        # a1, a2: angular accelerations of links 1 and 2
        # g1: gravity term
        # c1 = j1*a1 + mu1*g1 + d1*(2*w1 - w2) + dot(b1, u) + eps
        # Initially damping was d11*w1 + d12*w2 but later found out it always
        # had this structure
        c1 = np.cos(tt21)*a2 - np.sin(tt21)*w2**2
        g1 = g/l1*np.sin(tt1)
        
        x1 = (a1, g1, 2*w1 - w2)
        x1 = np.stack(x1, axis=-1)
        x1 = np.concatenate((x1, -u), axis=-1)
        X1 = x1.reshape((-1, x1.shape[-1]))
        
        coefs1,_,_,_ = lstsq(X1, -c1.reshape(-1))
        #c1h = -np.tensordot(x1, coefs1, axes=(-1,0))
        j1 = coefs1[0]
        mu1 = coefs1[1]
        d1 = coefs1[2]
        b1 = coefs1[3:]

        #% Estimate parameters of 2nd joint equations of motion
        # model is the following 
        # cg2: coriollis + centrifugal + gravity term
        # a1, a2: angular accelerations of links 1 and 2
        # cg2 = j2*a2 + d2*(w2 - w1) + dot(b2, u) + eps
        # Actually looks like we have d1 = d2 but I didn't enforce it here
        cg2 = np.cos(tt21)*a1 + np.sin(tt21)*w1**2 + g/l1*np.sin(tt2)
        
        x2 = (a2, w2 - w1)
        x2 = np.stack(x2, axis=-1)
        x2 = np.concatenate((x2, -u), axis=-1)

        X2 = x2.reshape((-1, x2.shape[-1]))
        
        coefs2,_,_,_ = lstsq(X2, -cg2.reshape(-1))
        #cg2h = -np.tensordot(x2, coefs2, axes=(-1,0))
        j2 = coefs2[0]
        d2 = coefs2[1]
        b2 = coefs2[2:]

        #% stack rows of input matrix
        B = np.stack((b1, b2), axis=0)
        
        # Do a "polar" decomposition of B into a symmetric and an orthogonal part
        # B = Bs@Br
        # Right multiply by the projection matrix onto the range of u
        U, s, Vt2 = np.linalg.svd(B)
        Bs = (U*s)@U.T
        Br = U@Vt2@Vt
        B = B@Vt

        results[beetle][input_var] = {
            'l1': l1,
            'l2': l2,
            'j1': j1,
            'j2': j2,
            'mu1': mu1,
            'd1': d1,
            'd2': d2,
            'B': B,
            'Bs': Bs,
            'Br': Br,
            }

#%% Save results
import pickle

with open(OUTPUT_FILE, 'wb') as file:
    pickle.dump(results, file)
