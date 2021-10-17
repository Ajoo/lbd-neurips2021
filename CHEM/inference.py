#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Estimate Parameters of Chemical System.

@author: ajoo
"""
import pandas as pd
import numpy as np
from scipy.optimize import minimize, least_squares

import jax.numpy as jnp
from jax import jit, vmap
from jax.experimental.ode import odeint
from jax.config import config
config.update("jax_enable_x64", True)

from functools import partial

from utils import load_systems

import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

from matplotlib import rcParams, cycler
cmap = plt.get_cmap('tab20')
rcParams['axes.prop_cycle'] = cycler(color = [cmap(k) for k in np.linspace(0, 1, 20)])

#%% Load training data

STRUCTURE_INFERENCE_RESULTS = 'structure_inference_results.pickle'
INFERENCE_RESULTS = 'inference_results.pickle'

t, x, u = load_systems()

#%% Specify the reactions infered in the previous step

reagents_ = [[5], 
             [8], #[8],
             [9], 
             [11], 
             [0, 14], #[0, 14],
             [1, 13],
             [2, 10], 
             [3, 4], 
             [6, 12], 
             [7, 14]]
products_ = [[14], 
             [1, 13], #[8, 9],
             [6, 12], 
             [14], 
             [8], #[0, 14, 8],
             [11],
             [5], 
             [0], 
             [7], 
             [9]] #[9, 14, 10]

def get_reactions(species, d=15):
    S = np.zeros((len(species), d))
    for i, s_ in enumerate(species):
        for s in s_:
            S[i, s] += 1
    return S

reagents = get_reactions(reagents_)
products = get_reactions(products_)

num_reactions = reagents.shape[0]
one_reagent = np.sum(reagents, -1) == 1
two_reagents = np.sum(reagents, -1) == 2
num_one_reagent = np.sum(one_reagent)
num_two_reagents = np.sum(two_reagents)

#%% Select a subset of the reactions
# This was only used during an exploration phase to fit smaller (approximately)
# separable subparts of the system instead of all the reactions at once
# It can be safely skipped

reactions_mask = list(range(len(reagents_)))
z_mask = np.unique(np.sum(np.array(reagents_, dtype=object)[reactions_mask]))
reagents = reagents[reactions_mask][:,z_mask]
products = products[reactions_mask][:,z_mask]

x = x[:, z_mask]
num_reactions = len(reactions_mask)

#%%  Define Functions to Simulate Dynamics

# dynamics without input (solving only for t=[t[3], 80])
@jit
def autonomous_flow(z, _, k):
    rates = k*jnp.prod(z**reagents, axis=-1)
    return jnp.dot(rates, products - reagents)


def autonomous_solver(z0, k):
    return odeint(autonomous_flow, z0, t[3:], k)

# vmap over instances and over systems to be able to simulate all at once
instances_autonomous_solver = jit(vmap(autonomous_solver, (-1, None), -1))
systems_autonomous_solver = jit(vmap(instances_autonomous_solver, (-1, -1), -1))

# dynamics with input (solving only for t=[0, t[3]])
@jit
def flow(z, _, k, Bu=0.0):
    z = jnp.maximum(0, z) # A "hack" to deal with negative concentrations
    rates = k*jnp.prod(z**reagents, axis=-1)
    dz = jnp.dot(rates, products - reagents)
    #if Bu is not None:
    dz += Bu
    return dz

@jit
def solver(z0, k, B, u):
    Bu = jnp.dot(B, u)
    Z1 = odeint(flow, z0, t[:4], k, Bu)
    return jnp.maximum(0, Z1)

# vmap over instances and over systems to be able to simulate all at once
instances_solver = jit(vmap(solver, (-1, None, None, -1), -1))
systems_solver = jit(vmap(instances_solver, (-1, -1, None, -1), -1))

# solve full system (for t=[0, 80])
@jit
def full_instances_solver(z0, k, B, u):
    Z1 = instances_solver(z0, k, B, u)
    Z2 = instances_autonomous_solver(Z1[-1], k)
    return jnp.concatenate((Z1, Z2[1:]), 0)

@jit
def full_systems_solver(z0, k, B, u):
    Z1 = systems_solver(z0, k ,B, u)
    Z2 = systems_autonomous_solver(Z1[-1], k)
    return jnp.concatenate((Z1, Z2[1:]), 0)

# define least squares problem to estimate all parameters of the system
def make_problem(x, k0, u=None, B0=None, z0=None, identify=set('k'), only_autonomous=True, only_forced=False):
    assert not (only_autonomous and only_forced), 'only_autonomous incompatible with only_forced'
    assert u is not None or only_autonomous, 'if not only_autonomous input must be provided'
    
    ndim = x.ndim
    if z0 is None: # No provided initial condition estimate
        z0 = jnp.maximum(0, x[0])
    if B0 is None: # No provided initial input matrix estimate
        B0 = np.zeros((x.shape[1], 8))

    autonomous_solver = systems_autonomous_solver if ndim==4 else instances_autonomous_solver
    
    # define residuals function as the difference between the simulated
    # trajectory and measurements
    if only_autonomous:
        def residuals(params):
            z, k, _ = get_params(params)
            Z = autonomous_solver(z, k)    
            return jnp.reshape(Z - x, -1)         
    else:
        solver = systems_solver if ndim==4 else instances_solver
        def residuals(params):
            z, k, B = get_params(params)
            Z1 = solver(z, k, B, u)
            if only_forced:
                r = Z1 - x[:4]
                return jnp.reshape(r, -1)
                
            z1 = jnp.maximum(0, Z1[-1])
            Z2 = autonomous_solver(z1, k)
            Z = np.concatenate((Z1, Z2[1:]), 0)
            r = Z - x
    
            return jnp.reshape(r, -1)
  
    # helper function to retrieve system parameters
    def get_params(params):
        i = 0
        if 'k' in identify:
            k = jnp.reshape(params[i:i+k0.size], k0.shape)
            i += k.size
        else:
            k = k0
        if 'B' in identify:
            B = jnp.reshape(params[i:i+B0.size], B0.shape)
            i += B.size
        else:
            B = B0
        if 'z' in identify:
            z = jnp.reshape(params[i:i+z0.size], z0.shape)
            i += z.size
        else:
            z = z0

        return z, k, B
    
    params = []
    if 'k' in identify:
        params.append(np.reshape(k0, -1))
    if 'B' in identify:
        params.append(np.reshape(B0, -1))
    if 'z' in identify:
        params.append(np.reshape(z0, -1))
    
    return residuals, get_params, np.hstack(params)


#%% Get initial estimates from previous script

with open(STRUCTURE_INFERENCE_RESULTS, 'rb') as file:
    structure_inference_results = pickle.load(file)
     
k0h = structure_inference_results['k']
# Average B estimate over all systems as it is supposed to be the shared
B0h = np.mean(structure_inference_results['B'], axis=0) 
z0h = np.maximum(0, x[0])

#%% Re-estimate reaction rates
# The first step is to take the rough initial rates estimates from the
# previous step and re-estimate them properly by solving a nonlinear least 
# squares problem. This can be done system by system since

kh = np.zeros(k0h.shape)
for system in range(12):
    residuals_fun, get_params, params0 = make_problem(
        x[3:,...,system], 
        k0h[...,system])
    
    result = least_squares(
        residuals_fun, 
        np.maximum(0, params0), #, 
        method='trf', #'lm',
        bounds=(0, np.inf),
        verbose=2)

    kh[...,system] = result.x
#%% Estimate initial conditions
# Since the measurements contain noise, using max(0, x[3]) is not really the
# true initial condition of the autonomous part of the simulation. 
# Here we let the optimizer also solve for the x[3] that minimizes the least
# square error for each simulation as this should lead to a better estimate
# of the reaction rates
# Note: This step can likely be skipped and we can proceed directly to estimating 
# the input matrix since it takes a while
    
z1h = np.zeros_like(x[3]) 
for system in range(12):
    residuals_fun, get_params, params0 = make_problem(
        x[3:,...,system], 
        kh[...,system],
        identify={'k', 'z'}
        )
    
    result = least_squares(
        residuals_fun, 
        np.maximum(0, params0), #, 
        method='trf', #'lm',
        bounds=(0, np.inf),
        verbose=2)

    z1h[...,system], kh[...,system], _ = get_params(result.x)
    
#%% Simulate Autonomous System  
    
Zh = systems_autonomous_solver(z1h, kh)

#%% Plot Results

# Choose a system and an instance to plot
system = 0
instance = 3

f, ax = plt.subplots(2,1)
ax[0].plot(t[3:], x[3:, :, instance, system])
ax[1].plot(t[3:], Zh[..., instance,system])
ax[1].set_ylim(ax[0].get_ylim())
ax[0].legend(np.arange(15), bbox_to_anchor=(1, 1.05))
plt.subplots_adjust(right=0.8)
#%% Plot Reaction Rates

plt.figure()
plt.plot(kh, 'o')
plt.legend(np.arange(12), bbox_to_anchor=(1, 1.05), title='System')
plt.subplots_adjust(right=0.8)
plt.xlabel('Reactions')
plt.ylabel('Rates')

# There seems to be some pattern here...

#%% Save intermediate results

inference_results = {
    'reagents': np.array(reagents_, dtype=object),
    'products': np.array(products_, dtype=object),
    'k': kh,
    'z1': z1h,
    }
with open(INFERENCE_RESULTS, 'wb') as file:
    pickle.dump(inference_results, file)

#%% Estimate Input Matrix
# Estimate only B keeping the estimated reaction rates fixed and with naive
# z0 estimate max(0, x[0])
# We could even use only the forced part of the simulations (i.e. from [0, t[3]])
# by setting only_forced to True

residuals_fun, get_params, params0 = make_problem(x, kh, u, B0h, 
                                                  only_autonomous=False,
                                                  only_forced=False,
                                                  identify={'B'})
result = least_squares(
        residuals_fun, 
        params0, #, 
        method='trf', #'lm',
        verbose=2)

_, _, Bh = get_params(result.x)
#%% Plot B coefficients
# The sparseness pattern of Bh is very obvious now even though the whitepaper
# didn'tmention it.  
# First 4 inputs each affects 2 species with a coefficient of 1
# Second 4 inputs each affects 2 species with a coefficient of 0.25

plt.figure()
plt.imshow(np.abs(Bh))
plt.ylabel('Species')
plt.xlabel('Inputs')

# Round it to nearest 0.05 to get rid of the small noise, not that it'll make
# much of a difference
Bh = np.round(Bh/0.05)*0.05
#%% Fit the entire simulations including the forced part
# Re-estimate the initial conditions now to be the initial conditions of the
# problem and not at t[3]
# Again, this is probably unnecessary for the purpose that we want but might
# as well...

for system in range(12):
    print('System', system)
    residuals_fun, get_params, params0 = make_problem(
        x[...,system], 
        kh[...,system],
        u[...,system],
        Bh,
        identify={'k', 'z'},
        only_autonomous=False,
        )
    
    result = least_squares(
        residuals_fun, 
        np.maximum(0, params0), 
        method='trf', #'lm',
        bounds=(0, np.inf),
        verbose=2)

    z0h[...,system], kh[...,system], _ = get_params(result.x)
    
#%% Simulate full system

Zh = full_systems_solver(z0h, kh, Bh, u)

#%% Plot Results

# Choose a system and an instance to plot
system = 1
instance = 11

f, ax = plt.subplots(2,1)
ax[0].plot(t, x[..., instance, system])
ax[1].plot(t, Zh[..., instance,system])
ax[1].set_ylim(ax[0].get_ylim())
ax[0].legend(np.arange(15), bbox_to_anchor=(1, 1.05))
plt.subplots_adjust(right=0.8)

#%% Save Final Results

inference_results = {
    'reagents': np.array(reagents_, dtype=object),
    'products': np.array(products_, dtype=object),
    'k': kh,
    'B': Bh,
    'z1': z1h,
    'z0': z0h,
    }
with open(INFERENCE_RESULTS, 'wb') as file:
    pickle.dump(inference_results, file)

