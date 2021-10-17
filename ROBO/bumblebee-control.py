#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Designs the controllers for the bumblebee robots.

@author: ajoo
"""
import pandas as pd
import numpy as np

SAVE_FILENAME = 'submissions/submission/{}-bumblebee.txt'
RHO = 0.1

bumblebees = ['great-bipedal', 'great-impartial', 'great-piquant', 'great-proficient',
              'lush-bipedal', 'lush-impartial', 'lush-piquant', 'lush-proficient']

g = np.array([0, -9.81])

#%% Load system identification results
import pickle

with open('bumblebee-results.pickle','rb') as file:
    results = pickle.load(file)

#%% Design affine LQR controller for each system
from control import clqra

A = np.zeros((4,4))
A[[0, 1], [2, 3]] = 1
B = np.zeros((4, 2))
c = np.zeros(4)
c[2:4] = g
  
for bumblebee in bumblebees:
    filename = SAVE_FILENAME.format(bumblebee)
    
    # Overwrite B matrix for the particular robot
    B[2:4] = results[bumblebee]['Bs']
    
    # Design continuous LQR (for affine system)
    K, k, P, p, sclose = clqra(A, B, c, q=[1, 1, 0, 0], rho=RHO)
    
    # Left multiply by orthogonal part of input matrix transposed
    Br = results[bumblebee]['Br']
    np.savetxt(filename, Br.T@np.c_[K, k])

