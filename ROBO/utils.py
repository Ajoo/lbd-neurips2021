#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 15:06:22 2021

@author: ajoo
"""
import pandas as pd
import numpy as np
import itertools as it

TRAIN_FILENAME = 'training_trajectories/{prefix}-{system}_{instance:02d}.csv'

def load_train(prefix, system, instances=range(50)):
    X, dX, U = [], [], []
    for instance in instances:
        t, x, dx, u = load_instance(prefix, system, instance)
        X.append(x)
        dX.append(dx)
        U.append(u)
    return t, np.stack(X, -1), np.stack(dX, -1), np.stack(U, -1)
        
        
def load_instance(prefix, system, instance):
    data = pd.read_csv(TRAIN_FILENAME.format(
        prefix=prefix, system=system, instance=instance))
    U = [col for col in data.columns if col.startswith('U')]
    X = [col for col in data.columns if col.startswith('X')]
    Y = [col for col in data.columns if col.startswith('Y')]
    dX = [col for col in data.columns if col.startswith('dX')]
    dY = [col for col in data.columns if col.startswith('dY')]
    
    X = list(it.chain.from_iterable(zip(X, Y)))
    dX = list(it.chain.from_iterable(zip(dX, dY)))
    
    t = data.t.values
    nt = len(t)
    x = data[X].values.reshape((nt, -1, 2))
    dx = data[dX].values.reshape((nt, -1, 2))
    u = data[U].values
    return t, x, dx, u