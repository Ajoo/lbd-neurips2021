#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Estimates the butterfly model parameters.

The results are saved in a pickle file.

@author: ajoo
"""
import pandas as pd
import numpy as np

from utils import load_train
from dynamics import to_butterfly_joint_space, remove_nullspace

from numpy.linalg import lstsq, svd
from collections import defaultdict

#%%
OUTPUT_FILE = 'butterfly-results.pickle'

butterfly_variations = ['talented', 'thoughtful']
input_variations = ['antique', 'ruddy', 'steel', 'zippy']

h = 0.01
g = 9.81

results = defaultdict(dict)
for butterfly in butterfly_variations:
    for input_var in input_variations:
        print('Identifying {}-{}'.format(butterfly, input_var))
        
        # Load all training trajectories for this variant
        t, x, v, u = load_train('{}-{}'.format(butterfly, input_var), 'butterfly')

        # Compute link lengths and
        # joint space angles, angular velocities and angular accelerations
        l1, l2, l3, tt, w, a = to_butterfly_joint_space(x, v, h)
        tt1, tt2, tt3 = tt[:,0], tt[:,1], tt[:,2]
        tt21, tt31, tt32 = tt2 - tt1, tt3 - tt1, tt3 - tt2
        w1, w2, w3 = w[:,0], w[:,1], w[:,2]
        a1, a2, a3 = a[:,0], a[:,1], a[:,2]
        
        # Ignore input subspace that is unexplored in training trajectories
        u = u[:-1]
        u, Vt, _ = remove_nullspace(u, 3)
        u = np.swapaxes(u, 1, -1)

        # Estimate parameters of 3rd joint equations of motion
        # cg3: coriollis + centrifugal + gravity term for 3rd link
        cg3 = l1*(np.cos(tt31)*a1 + np.sin(tt31)*w1**2) + \
              l2*(np.cos(tt32)*a2 + np.sin(tt32)*w2**2) + g*np.sin(tt3)
        x3 = (a3, w3)
        x3 = np.stack(x3, axis=-1)
        x3 = np.concatenate((x3, -u), axis=-1)
        X3 = x3.reshape((-1, x3.shape[-1])) # regressor
        coefs3,_,_,_ = lstsq(X3, -cg3.reshape(-1))
        #cg3h = -np.tensordot(x3, coefs3, axes=(-1,0))

        # Estimate parameters of 2nd joint equations of motion
        # cg2: part of coriollis + centrifugal + gravity term for link 2
        # c23: part of coriollis + centrifugal term for link 2
        cg2 = l1*(np.cos(tt21)*a1 + np.sin(tt21)*w1**2) + g*np.sin(tt2)
        c23 = l2*(np.cos(tt32)*a3 - np.sin(tt32)*w3**2)
        x2 = (a2, c23, w2)
        x2 = np.stack(x2, axis=-1)
        x2 = np.concatenate((x2, -u), axis=-1)
        X2 = x2.reshape((-1, x2.shape[-1])) # regressor
        coefs2,res,_,_ = lstsq(X2, -cg2.reshape(-1))
        #cg2h = -np.tensordot(x2, coefs2, axes=(-1,0))

        # Estimate parameters of 1st joint equations of motion
        # g1: gravity term for link 1
        # c12, c23: coriollis and centrifugal terms for link 1
        g1 = g*np.sin(tt1)
        c12 = l1*(np.cos(tt21)*a2 - np.sin(tt21)*w2**2)
        c13 = l1*(np.cos(tt31)*a3 - np.sin(tt31)*w3**2)
        x1 = (a1, c12, c13, w1)
        x1 = np.stack(x1, axis=-1)
        x1 = np.concatenate((x1, -u), axis=-1)
        X1 = x1.reshape((-1, x1.shape[-1]))
        coefs1,res,_,_ = lstsq(X1, -g1.reshape(-1))
        #g1h = -np.tensordot(x1, coefs1, axes=(-1,0))

        # Consolidate j12, j13 and j23 estimates
        # since they have to satisfy j13 = j12*j23
        # This small fixed point algorithm should solve
        # min (j12 - j12h)**2 + (j13 - j13h)**2 + (j23 - j23h)**2
        # subject to the above constraint
        # Note: this part is probably totally unnecessary
        j23 = coefs2[1]
        j12, j13 = coefs1[1], coefs1[2]
        j12h, j23h = j12, j23
        for i in range(10):
            #print(j12h, j23h)
            j12h = (j12 + j23h*j13)/(1 + j23h**2)
            j23h = (j23 + j12h*j13)/(1 + j12h**2)
        j13h = j12h*j23h

        # Retrieve the remaining coefficients and normalize them as appropriate
        j1, j2, j3 = coefs1[0], coefs2[0]*j12h, coefs3[0]*j13h
        jd = np.array([j1, j2, j3])
        
        d1 = coefs1[3]
        d2 = coefs2[2]*j12h
        d3 = coefs3[1]*j13h
        d = np.array([d1, d2, d3])
        
        b1 = coefs1[4:]
        b2 = coefs2[3:]*j12h
        b3 = coefs3[2:]*j13h
        
        #% stack rows of input matrix
        B = np.stack((b1, b2, b3), axis=0)
        U, s, Vt2 = np.linalg.svd(B)
        Bs = (U*s)@U.T
        Br = U@Vt2@Vt
        B = B@Vt
        
        # Do a "polar" decomposition of B into a symmetric and an orthogonal part
        # B = Bs@Br
        # Right multiply by the projection matrix onto the range of u        
        results[butterfly][input_var] = {
            'l1': l1,
            'l2': l2,
            'l3': l3,
            'j': jd,
            'j12': j12h,
            'j13': j13h,
            'j23': j23h,
            'd': d,
            'B': B,
            'Bs': Bs,
            'Br': Br,
            }

#%%
import pickle

with open(OUTPUT_FILE,'wb') as file:
    pickle.dump(results, file)
    