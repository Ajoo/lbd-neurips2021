#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimize Controls for Chemical System.

@author: ajoo
"""
import pandas as pd
import numpy as np
from scipy.optimize import minimize, least_squares
from scipy.stats import qmc

import jax
from jax import jit, vmap
import jax.numpy as jnp
from jax.experimental.ode import odeint
from functools import partial
from jax.config import config
config.update("jax_enable_x64", True)

from functools import partial
import pickle

from utils import load_test, make_submission

import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

from matplotlib import rcParams, cycler
cmap = plt.get_cmap('tab20')
rcParams['axes.prop_cycle'] = cycler(color = [cmap(k) for k in np.linspace(0, 1, 20)])
#%%
INFERENCE_RESULTS = 'inference_results.pickle'

# Load inference results
with open(INFERENCE_RESULTS, 'rb') as file:
    inference_results = pickle.load(file)

k = inference_results['k']
B = inference_results['B']

reagents_ = inference_results['reagents']
products_ = inference_results['products']

# Load test
x0, target, t = load_test()
x0 = np.maximum(0.0, x0)

#%% Get Chemical Reactions

def get_reactions(species, d=15):
    S = np.zeros((len(species), d))
    for i, s_ in enumerate(species):
        for s in s_:
            S[i, s] += 1
    return S

reagents = get_reactions(reagents_)
products = get_reactions(products_)

num_reactions = reagents.shape[0]

#%% Define solver

# This is essentially the same as before but with an added integrative state
# that computes the loss function
@jit
def autonomous_flow(z, t, k, target):
    rates = k*jnp.prod(z[:-1]**reagents, axis=-1)
    rates = jnp.dot(rates, products - reagents)
    cost_rate = jnp.where(t > 40.0, (z[-2] - target)**2, 0.0)
    return jnp.concatenate((rates, cost_rate[None]))

@jit
def autonomous_solver(t, z0, k, target):
    return odeint(autonomous_flow, z0, t, k, target)

@jit
def flow(z, _, k, Bu=0.0): #=None):
    z = jnp.maximum(0, z)
    rates = k*jnp.prod(z**reagents, axis=-1)
    dz = jnp.dot(rates, products - reagents)
    dz += Bu
    return dz

@jit
def solver(tu, t, z0, k, B, target, u):
    Bu = jnp.dot(B, u)
    Z1 = jnp.maximum(0, odeint(flow, z0, tu, k, Bu))
    z1 = jnp.concatenate((Z1[-1], np.zeros(1)))
    Z2 = odeint(autonomous_flow, z1, t, k, target)
    cost = jnp.concatenate((jnp.zeros(4), Z2[1:,-1]))
    return jnp.concatenate((Z1, Z2[1:,:-1]), 0), cost

# Choose times at which solution should be computed
# These are set as the times where a discontinuity in the dynamics
# exists
Tu = np.array([0.0, t[3]])
T = np.array([t[3], 40.0, 80.0])

@jit
def loss(z0, k, B, target, u):
    Bu = jnp.dot(B, u)
    z1 = odeint(flow, z0, Tu, k, Bu)[-1]
    z1 = jnp.maximum(0.0, z1)
    z1 = jnp.concatenate((z1, np.zeros(1)))
    jz = odeint(autonomous_flow, z1, T, k, target)[-1, -1]
    return jnp.sqrt(jz/40.0) + jnp.sqrt(jnp.dot(u, u)/8.0)/20.0

# vmap it over instances and systems so that we get a simple way to compute 
# the loss for everything at the end    
instances_loss = jit(vmap(loss, (-1, None, None, -1, -1), -1))
systems_loss = jit(vmap(instances_loss, (-1, -1, None, -1, -1), -1))


#%%  Solve Control Problem

# Sample initial guesses using a Latin Hypercube Sampler
sampler = qmc.LatinHypercube(d=8, seed=123)
U0 = 20*sampler.random(10) - 10

U = np.tile(U0, (12, 50, 1, 1)) # array to hold optimized inputs
L = np.zeros((12, 50, U0.shape[0])) # array to hold losses
for system in range(12):
    for instance in range(50):
        print('System {}, instance {}'.format(system, instance))
        for i, u0 in enumerate(U0):
            loss_func = partial(loss,
                x0[..., instance, system], 
                k[..., system], 
                B, 
                target[instance, system]
                )
            
            result = minimize(
                loss_func, 
                u0,
                bounds=[(-10.0, 10.0)]*8)
            U[system, instance, i] = result.x
            L[system, instance, i] = result.fun

#%% Save Results

np.save('inputs.npy', U)
#%% Evaluate Expected Loss

# Select best solution for each system and instance
idx = np.argmin(L, -1)
u = np.take_along_axis(U, idx[...,None,None], axis=-2)[...,0,:]
u = np.swapaxes(u, 0, -1)

# Evaluate solution
l = systems_loss(x0, k, B, target, u)

print('Expected loss:', np.mean(l))
#%% Create Final Submission

make_submission(u)