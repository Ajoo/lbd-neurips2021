# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
from jax import jit, vmap
from jax.lax import scan


def euler(flow, h, num_steps=10):
    '''
    Creates simulator for a dynamical system with exogenous input by Euler 
    integration with multiple euler steps per input
    '''
    h_step = h/num_steps
    
    @jit
    def step_flow(x, u):
        dx, y = flow(x, u)
        return x + h_step*dx, y
    
    @jit
    def _solver(x, u):
        return scan(step_flow, x, u)
    
    @jit
    def solver(x0, u):
        u = jnp.repeat(u, num_steps, axis=0)[:-num_steps]
        return _solver(x0, u)[1][::num_steps]

    return solver


def simulate(flow, controller, h, num_steps=10):
     '''
    Creates simulator for closed loop system comprisin a dynamical system with 
    exogenous input and a controller. Does multiple Euler steps per
    controller step
    '''
    h_step = h/num_steps
    
    @jit
    def step_flow(x, u):
        dx, y = flow(x, u)
        return x + h_step*dx, y
    
    @jit
    def step(x, ref):
        x_sys, x_ctrl = x
        u, y_ctrl, x_ctrl= controller(x_sys, ref, x_ctrl)
        u = jnp.tile(u, (num_steps, 1))
        x_sys, y = scan(step_flow, x_sys, u)
        return (x_sys, x_ctrl), (y[-1], y_ctrl)
    
    @jit
    def simulator(x0, ref, x0_ctrl=None):
        return scan(step, (x0, x0_ctrl), ref)[1]

    return simulator

    
#%% Old Functions
# These functions were helper functions for exploration and debugging
# when I was exploring different solutions

@jit
def get_x1_ref(x, l1, l2):
    d = jnp.sqrt(jnp.sum(x**2))
    l = (l1**2 - l2**2 + d**2)/2/d
    l = jnp.clip(l, -l1, l1)
    h = jnp.sqrt(l1**2 - l**2)
    
    center = x*l/d
    side = jnp.array([x[1], -x[0]])*h/d

    return jnp.stack((center + side, center - side), axis=0)

vectorized_get_x1_ref = jit(vmap(get_x1_ref, (0, None, None), 0))

def get_x1_ref_histories(x2ref, l1, l2):
    nt = x2ref.shape[0]
    x1ref = vectorized_get_x1_ref(x2ref, l1, l2)
    dx1ref = x1ref[1:,:,None,:] - x1ref[:-1,None,:,:]
    dx1ref2 = np.sum(dx1ref**2, -1)
    idx = np.mod(np.cumsum(np.argmin(dx1ref2, -1)[:,0]), 2)
    idx = np.insert(idx, 0, 0)
    return np.stack((x1ref[np.arange(nt), idx, :], x1ref[np.arange(nt), 1-idx, :]), axis=1)
    
import os
import imageio
import webbrowser


def make_gif(x, ref, filename='animation.gif'):
    x = np.insert(x, 0, 0, axis=1)
    
    nt = x.shape[0]
    tempname = 'temp.png'
    with imageio.get_writer(filename, mode='I') as writer:
        for i in range(0,nt,2):
            f, ax = plt.subplots()
            ax = plot_reference(ref, ax=ax)
            ax.plot(ref[i,0], ref[i,1], 'og')
            ax.plot(x[0,:,0], x[0,:,1], 'k-o')
            ax.plot(x[:i,-1,0], x[:i,-1,1], color='orange')
            ax.plot(x[i,:,0], x[i,:,1], 'b-o')
            f.savefig(tempname)
            plt.close(f)
            image = imageio.imread(tempname)
            writer.append_data(image)
            os.remove(tempname)
    
    webbrowser.open(filename)

def plot_reference(x2ref, x1ref=None, l1=None, l2=None, ax=None):
    ax = plot2(x2ref, '--g', ax=ax)
    ax.plot(x2ref[-1, 0], x2ref[-1, 1], '*g')
    if x1ref is None:
        return ax
    if x1ref == "compute":
        x1ref = get_x1_ref_histories(x2ref, l1, l2)
        ax.plot(x1ref[:,0,0], x1ref[:,0,1], '--r')
        ax.plot(x1ref[:,1,0], x1ref[:,1,1], '--m')
    ax.plot(x1ref[:,0], x1ref[:,1], '--r')
    return ax


def plot2(x, *args, **kwargs):
    ax = kwargs.pop('ax', None)
    if ax is None:
        _, ax = plt.subplots()
    ax.plot(x[:,0], x[:,1], *args, **kwargs)
    ax.axis('square')
    ax.set_xlim([-3, 3])
    ax.set_ylim([-3, 3])
    ax.plot(0, 0, 'ok')
    return ax
