#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions to Load train and test data and create the final submission.

@author: ajoo
"""
import pandas as pd
import numpy as np

from functools import partial

TRAIN_FILENAME ='CHEM_trainingdata/system_{:02d}_instance_{:02d}.csv'
TEST_FILENAME = 'CHEM_starter_kit/submission_template.csv'

X = ['X{:d}'.format(i) for i in range(1, 15)] + ['Y']
U = ['U{:d}'.format(i) for i in range(1, 9)]

def load_system(system, instances=range(20)):
    df = tuple(pd.read_csv(TRAIN_FILENAME.format(system, i)).drop('ID', 1) for i in instances)
    df = pd.concat(df, axis=0, keys=instances, names=['instance'])

    t = df.loc[0, 't']
    assert (df['t'].unstack() == t).all().all()
    
    u = df[U].unstack(0).iloc[0].unstack()
    
    x = df[X]
    x = x.unstack(0).values.reshape(len(t), len(X), -1)
    return t.values, x, u.values

def load_systems(systems=range(1, 13), instances=range(20)):
    data = tuple(load_system(system, instances) for system in systems)
    t, x, u = map(partial(np.stack, axis=-1), zip(*data))
    
    return t[...,0], x, u

def load_test():
    t = pd.read_csv(TRAIN_FILENAME.format(1, 0))['t'].values
    test = pd.read_csv(TEST_FILENAME)
    x0 = test[X].values.reshape((12,-1,15))
    target = test.target.values.reshape((12,-1))
    return np.swapaxes(x0, 0, -1), target.T, t
    
def make_submission(u, filename='submission.zip'):
    test = pd.read_csv(TEST_FILENAME)
    test.loc[:,U] = np.swapaxes(u, 0, -1).reshape((-1, 8))
    
    test.to_csv(filename, index=None, compression={'method':'zip', 'archive_name':'submission.csv'})