#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Infer the Structure of the Chemical System.

@author: ajoo
"""
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from sklearn.linear_model import LassoCV, LassoLarsCV, RidgeCV, ElasticNetCV
from sklearn.model_selection import GroupKFold

from functools import partial, reduce
import operator as op

from utils import load_systems

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from matplotlib import rcParams, cycler
cmap = plt.get_cmap('tab20')
rcParams['axes.prop_cycle'] = cycler(color = [cmap(k) for k in np.linspace(0, 1, 20)])

#%% Define functions to perform structure inference

OUTPUT_FILE = 'structure_inference_results.pickle'

num_instances = 20
num_systems = 12
num_dims = 15
num_inputs = 8
num_quadratic = int(num_dims*(num_dims + 1)/2)

def estimate_derivative(x, t, windowlength, polyorder, Tu=3):
    '''
    Estimate the time derivative of the concentrations for a simulation 
    using finite differences and given that it is discontinuous at t = t[Tu]
    '''
    dt1 = savgol_filter(t[:Tu+1], windowlength, polyorder, 1)
    dt2 = savgol_filter(t[Tu:], windowlength, polyorder, 1)
    dt = np.concatenate((dt1, dt2), axis=0)
    #dt = np.insert(dt, Tu, dt[Tu])
    dx1 = savgol_filter(x[:Tu+1], windowlength, polyorder, 1, axis=0)
    dx2 = savgol_filter(x[Tu:], windowlength, polyorder, 1, axis=0)#[1:]
    dx = np.concatenate((dx1, dx2), axis=0)
    
    return dx/dt[(slice(None),) + (None,)*(dx.ndim - 1)]

def add_discontinuity(x, T=3):
    '''
    Add a duplicate entry at position T over the first axis
    '''
    return np.insert(x, T, x[T], axis=0)

def make_input(u, nt=20, T=3):
    '''
    Reconstruct input (over time)
    '''
    uz = np.zeros((nt-T,) + u.shape)
    u = np.repeat(u[None], T+1, axis=0)
    return np.concatenate((u, uz), axis=0)

def _upper_triangular(n, k=0):
    for i in range(n):
        for j in range(i+k, n):
            yield i*n + j
def upper_triangular(n, k=0):
    '''
    Get indices for upper triangular part of a square matrix starting at
    a specified diagonal

    Parameters
    ----------
    n : dimension of square matrix
    k : statring diagonal

    Returns
    -------
    Tuple of indices into the upper triangular part

    '''
    return tuple(_upper_triangular(n, k))

def make_quadratic(x, k=0):
    '''
    Get all quadratic monomials

    Parameters
    ----------
    x : array of shape (nt, d, ...)
    k : first diagonal to include. 
        The default is 0 which includes all terms such as xi*xj for j >= i
        Set to 1 to ignore quadratic terms such as xi**2 (i.e., include only j > i)

    Returns
    -------
    array of shape (nt, (d-k)*(d-k+1)/2, ...)

    '''
    x2 = np.reshape(x[:,None]*x[:,:,None], (x.shape[0], -1) + x.shape[2:])
    # pick only upper diagonal terms
    return np.take(x2, upper_triangular(x.shape[1], k=k), axis=1)

# mapping between quadratic index term and the two reagents (0-indexed)
quadratic_indexes = np.array(upper_triangular(num_dims))
quadratic_reagents = np.stack((quadratic_indexes//num_dims, np.mod(quadratic_indexes, num_dims)), -1)

def get_groups(shape, group_dim=0):
    '''
    Define groups (for cross-validation) by a specified dimension of matrix of
    given shape
    '''
    if group_dim < 0:
        group_dim = len(shape) + group_dim
    group = np.arange(shape[group_dim])
    return np.broadcast_to(group[(slice(None),) + (None,)*(len(shape)-1 - group_dim)], shape)


def fit(X1, X2, U, y, 
        t_mask=slice(4, None), # by default use only autonomous part (where u=0)
        X1_mask=[list(range(num_dims)) for _ in range(num_dims)], # by default include all linear terms
        X2_mask=[list(range(num_quadratic)) for _ in range(num_dims)], # and quadratic...
        U_mask=[[] for _ in range(num_dims)], # ignore input by default (autonomous)
        model=LassoLarsCV):
    '''
    Parameters
    ----------
    X1 : array with dimensions (time, instances, systems, linear terms) of linear monomials 
    X2 : array with dimensions (time, instances, systems, quadratic terms) of quadratic monomials
    U : array with dimentions (time, instances, systems, inputs) of inputs
    y : array of dimensions () concentration rates
    t_mask : Mask of timesteps to consider. The default is slice(4, None).
    X1_mask : Mask of linear terms to consider for each rate equation
    X2_mask : Mask of quadratic terms to consider for each rate equation
    U_mask : Mask of inputs to consider for each rate equation
    model : CV model to fit. Default is LassoLarsCV

    Returns
    -------
    A1 : linear coeficients. Array of shape (num_systems, num_dims, num_linear_terms)
    A2 : quadratic coeficients. Array of shape (num_systems, num_dims, num_quadratic_terms)
    B : input coeficients. Array of shape (num_systems, num_dims, num_inputs)
    linear_terms : linear monomials considered
    quadratic_terms : quadratic monomials considered

    '''
    
    # find all linear and quadratic monomials used
    linear_terms = sorted(list(set(reduce(op.add, X1_mask))))
    quadratic_terms = sorted(list(set(reduce(op.add, X2_mask))))
    #print(linear_terms)
    #print(quadratic_terms)
    
    # number of linear and quadratic monomials used
    num_linear = len(linear_terms)
    num_quadratic = len(quadratic_terms)
    
    # initialize coeficients for input, linear and quadratic terms
    B = np.zeros((num_systems, num_dims, num_inputs))
    A1 = np.zeros((num_systems, num_dims, num_linear))
    A2 = np.zeros((num_systems, num_dims, num_quadratic))
    
    # vector for determining CV folds based on instances
    g = get_groups(X1[t_mask,:].shape[:-2], 1).reshape(-1)
    y = y[t_mask]

    # for each species fit the linear rate equation for each system
    for dim in range(num_dims):
        X = []
        
        # collect all terms present for this species' rate equation
        if len(U_mask[dim]) != 0:
            X.append(U[...,U_mask[dim]])
        if len(X1_mask[dim]) != 0:
            X.append(X1[...,X1_mask[dim]])
        if len(X2_mask[dim]) != 0:
            X.append(X2[...,X2_mask[dim]])
        X = np.concatenate(X, axis=-1)[t_mask] # regressor
 
        # for each system find the rate equation coeficients for this species
        for system in range(num_systems):
            Xsys = X[:, :, system].reshape((-1, X.shape[-1]))
            ysys = y[:, :, system, dim].reshape(-1)
            
            cv=GroupKFold(4).split(Xsys, ysys, g)
            reg = model(fit_intercept=False, cv=cv).fit(Xsys, ysys)
            
            idx = 0
            if len(U_mask[dim]) != 0:
                coef_idx = U_mask[dim]
                B[system, dim, coef_idx] = reg.coef_[idx:idx+len(coef_idx)]
                idx += len(coef_idx)
            if len(X1_mask[dim]) != 0:
                coef_idx = [linear_terms.index(term) for term in X1_mask[dim]]
                A1[system, dim, coef_idx] = reg.coef_[idx:idx+len(coef_idx)]
                idx += len(coef_idx)
            if len(X2_mask[dim]) != 0:
                coef_idx = [quadratic_terms.index(term) for term in X2_mask[dim]]
                A2[system, dim, coef_idx] = reg.coef_[idx:idx+len(coef_idx)]
                idx += len(coef_idx)
            
    return A1, A2, B, linear_terms, quadratic_terms

# The following are a set of heuristics to help in the structure identification
# The key is looking at the signs (and magnitudes) of the estimated coefficients


def find_single_reagent_consistent_signs(A1, linear_terms):
    '''
    Find single reagent reactions whose coefficients are consistent
    For a reaction:
    
        x_i -> ?? 
        
    The coefficient of x_i in the equation for the rate of species i should
    be negative.
    This lets us identify the reagents part of reactions that are plausibly 
    present in the data by checking what signs are consistent across the
    different systems.
    '''
    diagonal = A1[:, linear_terms, list(range(len(linear_terms)))]
    # number of systems where the estimated coefficient has the wrong and
    # the right signs
    incorrect = np.sum(diagonal > 0, 0)
    correct = np.sum(diagonal < 0, 0)
    
    # rank all the terms first by smallest number of incorrect coefficients
    # second by largest number of correct coefficients
    ranking_idx = np.lexsort((-correct, incorrect))
    return (
        ranking_idx, 
        np.asarray(linear_terms)[ranking_idx], 
        incorrect[ranking_idx], 
        correct[ranking_idx]
        )

def find_two_reagent_consistent_signs(A2, quadratic_terms):
    '''
    Find two reagent reactions whose coefficients are consistent
    For a reaction:
    
        x_i + x_j -> ?? 
        
    The coefficient of x_i*x_j in the equation for the rate of species i and j 
    should be negative.
    '''
    reagents = quadratic_reagents[quadratic_terms]
    reagent1 = A2[:, reagents[:,0], list(range(len(quadratic_terms)))]
    reagent2 = A2[:, reagents[:,1], list(range(len(quadratic_terms)))]
    
    incorrect1 = np.sum(reagent1 > 0, 0)
    correct1 = np.sum(reagent1 < 0, 0)
    
    incorrect2 = np.sum(reagent2 > 0, 0)
    correct2 = np.sum(reagent2 < 0, 0)
    
    incorrect = incorrect1 + incorrect2
    correct = correct1 + correct2
    
    # rank all the terms first by smallest number of incorrect coefficients
    # second by largest number of correct coefficients
    ranking_idx = np.lexsort((-correct, incorrect))
    return (
        ranking_idx, 
        np.asarray(quadratic_terms)[ranking_idx], 
        incorrect[ranking_idx], 
        correct[ranking_idx]
        )

def find_single_reagent_plausible_products(A1, linear_terms):
    A1 = A1.copy()
    A1[:, linear_terms, list(range(len(linear_terms)))] *= -1
    
    incorrect = np.sum(A1 < 0, 0)
    correct = np.sum(A1 > 0, 0)
    return correct, incorrect

def get_mask(condition, terms):
    return [list(np.asarray(terms)[np.where(c)]) for c in condition]


def find_two_reagent_plausible_products(A2, quadratic_terms):
    A2 = A2.copy()
    reagents = quadratic_reagents[quadratic_terms]
    A2[:, reagents[:,0], list(range(len(quadratic_terms)))] *= -1
    A2[:, reagents[:,1], list(range(len(quadratic_terms)))] *= -1
    
    incorrect = np.sum(A2 < 0, 0)
    correct = np.sum(A2 > 0, 0)
    return correct, incorrect

def plot_species(a1, a2, linear_terms, quadratic_terms):
    f, axes = plt.subplots(2, 1)
    axes[0].plot(linear_terms, a1.T, 'o')
    axes[1].plot(quadratic_terms, a2.T, '.-')
    return axes

def plot_reactions(A, terms, kind='scatter', title_func=None):
    axes = []
    for i, term in enumerate(terms):
        f, ax = plt.subplots(figsize=(10, 5))
        if kind=='scatter':
            ax.plot(A[...,i].T, 'o')
        else:
            pd.DataFrame(A[...,i]).plot(kind='box', ax=ax)
        if title_func is not None:
            ax.set_title(title_func(term))
        
        ax.set_xlabel('Species')
        axes.append(ax)
    return axes

def plot_single_reagent_reactions(A1, linear_terms, kind='scatter'):
    title_func = lambda term: 'Reagent: {}'.format(term)
    return plot_reactions(A1, linear_terms, kind=kind, title_func=title_func)
    
def plot_two_reagent_reactions(A2, quadratic_terms, kind='scatter'):
    title_func = lambda term: 'Reagents: {}, {}'.format(*quadratic_reagents[term])
    return plot_reactions(A2, quadratic_terms, kind=kind, title_func=title_func)

#%% Load all simulations for all systems 

t, x, u = load_systems()

# t: array of shape (nt,)
# x: array of shape (nt, d, num_instances, num_systems)
# u: array of shape (p, num_instances, num_systems) 
#    is the input provided to each (instance, system) from time 0 to t[3]

#%% Build least squares problem
dx_dt = estimate_derivative(x, t, 3, 1)
xd = add_discontinuity(x)
td = add_discontinuity(t)

y = np.moveaxis(dx_dt, 1, -1)
U = np.moveaxis(make_input(u), 1, -1)
X1 = np.moveaxis(xd, 1, -1)
X2 = np.moveaxis(make_quadratic(xd), 1, -1)


#%% Start by fitting a linear model including all possible linear and quadratic 
# monomials on the autonomous part of the training data

A1, A2, _, linear_terms, quadratic_terms = fit(X1, X2, U, y)
#%% Identify potential reactions by the least number of incorrect signs
# across systems

promising_linear_term_indexes, promising_linear_terms, incorrect1, correct1 = find_single_reagent_consistent_signs(A1, linear_terms)
promising_quadratic_term_indexes, promising_quadratic_terms, incorrect2, correct2 = find_two_reagent_consistent_signs(A2, quadratic_terms)

# cap the number of monomials to consider to those that have 0 incorrect
# coefficients for linear and 1 incorrect coefficient in quadratic
promising_linear_terms = promising_linear_terms[incorrect1 <= 0]
promising_quadratic_terms = promising_quadratic_terms[incorrect2 <= 1]

#%% Plotting the coefficients of each identified linear and quadratic terms
# for each concentration rate equation already gives us a glimpse of the reactions
# For monomials x_i we expect to see only negative coefficients on the eq.
# for species i and positive coefficients for the products of the reaction
#     (i) -> (j) + (k) + ...
# For monomials x_i*x_j we expect to see only negative coefficients on the 
# equations for species i and j and positive coefficients for the products k, l, ...
#     (i) + (j) -> (k) + (l) + ...
# Furthermore the magnitude of coefficients for the two reagents should be
# similar and for the products it should be a multiple of that (the stochiometry) 
# Note: I always refer to the species by 0-index instead of the 1-index used in 
# the whitepaper 

plot_single_reagent_reactions(
    A1[..., promising_linear_term_indexes], 
    promising_linear_terms, 
    kind='scatter')

plot_two_reagent_reactions(
    A2[..., promising_quadratic_term_indexes], 
    promising_quadratic_terms, 
    kind='scatter')

# We can already identify some likely reactions such as:
# (2) + (10) -> (5)
# (1) + (13) -> (11)

#%% Now, repeating the procedure keeping only the most promising monomials 
# gets rid of a lot of the noise and gives us a much clearer picture
# For now trying to identify the products of the reactions is difficult
# so keep every product as an option for every reaction
mask_X1 = [list(promising_linear_terms) for _ in range(num_dims)]
mask_X2 = [list(promising_quadratic_terms) for _ in range(num_dims)]

A1, A2, _, linear_terms, quadratic_terms = fit(X1, X2, U, y, slice(4, None), mask_X1, mask_X2)

promising_linear_term_indexes, promising_linear_terms, incorrect1, correct1 = find_single_reagent_consistent_signs(A1, linear_terms)
promising_quadratic_term_indexes, promising_quadratic_terms, incorrect2, correct2 = find_two_reagent_consistent_signs(A2, quadratic_terms)

promising_linear_terms = promising_linear_terms[incorrect1 <= 0]
promising_quadratic_terms = promising_quadratic_terms[incorrect2 <= 1]

promising_linear_term_indexes = promising_linear_term_indexes[incorrect1 <= 0]
promising_quadratic_term_indexes = promising_quadratic_term_indexes[incorrect2 <= 1]
#%%

plot_single_reagent_reactions(
    A1[..., promising_linear_term_indexes], 
    promising_linear_terms, 
    kind='scatter')

plot_two_reagent_reactions(
    A2[..., promising_quadratic_term_indexes], 
    promising_quadratic_terms, 
    kind='scatter')

#%% Finally, find products for each reaction

correct1, incorrect1 = find_single_reagent_plausible_products(
    A1[..., promising_linear_term_indexes], promising_linear_terms)

correct2, incorrect2 = find_two_reagent_plausible_products(
    A2[..., promising_quadratic_term_indexes], promising_quadratic_terms)

mask_X1 = get_mask((incorrect1 <= 1) & (correct1 >= 1), promising_linear_terms)
mask_X2 = get_mask((incorrect2 <= 1) & (correct2 >= 1), promising_quadratic_terms)

#%% Run it a couple more times!

for _ in range(2):
    A1, A2, _, linear_terms, quadratic_terms = fit(X1, X2, U, y, slice(4, None), mask_X1, mask_X2)
    
    promising_linear_term_indexes, promising_linear_terms, incorrect1, correct1 = find_single_reagent_consistent_signs(A1, linear_terms)
    promising_quadratic_term_indexes, promising_quadratic_terms, incorrect2, correct2 = find_two_reagent_consistent_signs(A2, quadratic_terms)
    
    promising_linear_terms = promising_linear_terms[incorrect1 <= 0]
    promising_quadratic_terms = promising_quadratic_terms[incorrect2 <= 1]
    
    promising_linear_term_indexes = promising_linear_term_indexes[incorrect1 <= 0]
    promising_quadratic_term_indexes = promising_quadratic_term_indexes[incorrect2 <= 1]
    
    correct1, incorrect1 = find_single_reagent_plausible_products(
        A1[..., promising_linear_term_indexes], promising_linear_terms)
    
    correct2, incorrect2 = find_two_reagent_plausible_products(
        A2[..., promising_quadratic_term_indexes], promising_quadratic_terms)

    mask_X1 = get_mask((incorrect1 <= 1) & (correct1 >= 1), promising_linear_terms)
    mask_X2 = get_mask((incorrect2 <= 1) & (correct2 >= 1), promising_quadratic_terms)
    
#%% Finally, estimate the input matrix B
# This cell is just informational as the results are not used


mask_B = [list(range(num_inputs)) for _ in range(num_dims)]
A1, A2, B, linear_terms, quadratic_terms = fit(X1, X2, U, y, slice(None), mask_X1, mask_X2, mask_B)
# Even though it looks like there is also a sparsity pattern, I didn't try to exploit that here

for i in range(num_dims):
    f, ax = plt.subplots()
    ax.plot(B[:,i].T, 'o')
    #pd.DataFrame(B[:,i]).plot(kind='box', ax=ax)
    ax.set_title('Species {}'.format(i))
    ax.set_xlabel('Input')

#%% 
# Now with the sparsity pattern hopefully correct we switch to a Ridge Regression
# to get an initial estimate of the coefficients for the next step

A1, A2, B, linear_terms, quadratic_terms = fit(X1, X2, U, y, slice(None), mask_X1, mask_X2, mask_B, model=RidgeCV)

#%% plot final set of reactions

plot_single_reagent_reactions(
    A1[..., promising_linear_term_indexes], 
    promising_linear_terms, 
    kind='scatter')

plot_two_reagent_reactions(
    A2[..., promising_quadratic_term_indexes], 
    promising_quadratic_terms, 
    kind='scatter')


#%% Print Reactions and get initial estimate of reaction rates 
# assuming that all stochiometries are 1 through inspection of the coefficient 
# plots (and experimenting with slight changes to structure in the next step)

print('Set of Reactions:')
k1 = np.zeros((num_systems, len(promising_linear_terms)))
k2 = np.zeros((num_systems, len(promising_quadratic_terms)))

I = np.argsort(promising_linear_terms)
for i, term in zip(promising_linear_term_indexes[I], promising_linear_terms[I]):
    reagent = term
    products = [i for i, s in enumerate(mask_X1) if i != term and term in s]
    print(reagent, '->', ', '.join([str(p) for p in products]))
    
    # Take mean coefficient over products and reagents
    k1[:,i] = -A1[:, reagent, i]
    k1[:,i] += np.sum(A1[:, products, i], 1)
    k1[:,i] /= len(products) + 1
    
I = np.argsort(promising_quadratic_terms)
for i, term in zip(promising_quadratic_term_indexes[I], promising_quadratic_terms[I]):
    reagent1, reagent2 = quadratic_reagents[term]
    products = [i for i, s in enumerate(mask_X2) if i != reagent1 and i != reagent2 and term in s]
    print(reagent1, ',', reagent2, '->', ', '.join([str(p) for p in products]))
    
    # Take mean coefficient over products and reagents
    k2[:,i] = - A2[:, reagent1, i] - A2[:, reagent2, i]
    k2[:,i] += np.sum(A2[:, products, i])
    k2[:,i] /= len(products) + 2

#%% Save Results

import pickle

results = {
    'k1': np.maximum(0, k1).T,
    'k2': np.maximum(0, k2).T,
    'k': np.maximum(0, np.concatenate((k1, k2), axis=-1)).T,
    'B': B,
    }

with open(OUTPUT_FILE, 'wb') as file:
    pickle.dump(results, file)
    