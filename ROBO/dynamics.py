# -*- coding: utf-8 -*-
import numpy as np
from numpy.linalg import svd

import jax.numpy as jnp
from jax import jit

g = 9.81

def remove_nullspace(u, num_dims):
    '''
    Finds an orthogonal basis of dimension num_dims for the input by keeping
    only the first num_dims vector of a singular value decomposition.
    This can be used to remove the linear subspace of the input space 
    that is "unused" by the controller since we can only estimate the effect of
    the control input on the subspace that is used.

    Parameters
    ----------
    u : Original input
    num_dims : Number of dimensions to keep

    Returns
    -------
    u_bar : Input in the chosen orthogonal basis
    Vt : Matrix whose rows are the vectors of the orthogonal basis. I.e.
         when the s[num_dims:] = 0 we have:
         u = Vt.T@u_bar
    s : singular values of u

    '''
    u = np.swapaxes(u, 1, -1)
    U, s, Vt = svd(u.reshape(-1, u.shape[-1]), full_matrices=False)
    u_bar = np.reshape(U[:,:num_dims]*s[:num_dims], (*u.shape[:-1], num_dims)).swapaxes(1, -1)
    Vt = Vt[:num_dims] # orthogonal basis for u
    return u_bar, Vt, s

def to_continuous(tt):
    '''
    Makes a sequence of angles with 2\pi jumps continuous
    '''
    dtt = np.diff(tt, axis=0)
    dtt = np.insert(dtt, 0, 0, axis=0)
    jumps = np.cumsum(-np.round(dtt/2/np.pi), axis=0)
    return tt + 2*np.pi*jumps

def to_polar(x, v):
    '''
    Convert a link position and velocity into ccw angle with vertical
    and angular speed
    '''
    tt = to_continuous(np.arctan2(x[:,0], -x[:,1]))
    w = np.sum(v**2, 1)/(x[:,0]*v[:,1] - x[:,1]*v[:,0])
    mask = np.all(v == 0, 1)
    w[mask] = 0.0
    return tt, w

def relative_to(tt, w, tt0, w0):
    return tt - tt0, w - w0

def rolling_mean(x):
    '''
    rolling mean over two consecutive samples
    '''
    return (x[:-1] + x[1:])/2

def to_beetle_joint_space(x, v, h=None):
    '''
    Convert from joint positions and velocities into link angles and
    angular rates for the beetle robot
    '''
    l1 = np.mean(np.sqrt(np.sum(x[:,1]**2, 1)))
    l2 = np.mean(np.sqrt(np.sum((x[:,0] - x[:,1])**2, 1)))
    
    tt1, w1 = to_polar(x[:,1], v[:,1])
    tt2, w2 = to_polar(x[:,0] - x[:,1], v[:,0] - v[:,1])
    
    tt = np.stack((tt1, tt2), axis=1)
    w = np.stack((w1, w2), axis=1)
    
    if h is None: # don't compute accelerations
        return tt, w
    
    a = np.diff(w, axis=0)/h
    tt = rolling_mean(tt)
    w = rolling_mean(w)
    
    return l1, l2, tt, w, a

def to_butterfly_joint_space(x, v, h=None):
    '''
    Convert from joint positions and velocities into link angles and
    angular rates for the butterfly robot
    '''
    l1 = np.mean(np.sqrt(np.sum(x[:,2]**2, 1)))
    l2 = np.mean(np.sqrt(np.sum((x[:,1] - x[:,2])**2, 1)))
    l3 = np.mean(np.sqrt(np.sum((x[:,0] - x[:,1])**2, 1)))
    l1, l2, l3 = np.round(l1/0.5)*0.5, np.round(l2/0.5)*0.5, np.round(l3/0.5)*0.5
    
    tt1, w1 = to_polar(x[:,2], v[:,2])
    tt2, w2 = to_polar(x[:,1] - x[:,2], v[:,1] - v[:,2])
    tt3, w3 = to_polar(x[:,0] - x[:,1], v[:,0] - v[:,1])
    
    tt = np.stack((tt1, tt2, tt3), axis=1)
    w = np.stack((w1, w2, w3), axis=1)
    
    if h is None: # don't compute accelerations
        return tt, w
    
    a = np.diff(w, axis=0)/h
    tt = rolling_mean(tt)
    w = rolling_mean(w)
    
    return l1, l2, l3, tt, w, a
    

@jit
def beetle_dynamics(params, x, u):
    '''
    Dynamics to simulate beetle robot
    '''
    l1, l2, j1, j2, mu1, d, B = params
    tt1, tt2, w1, w2 = x
    tt21 = tt2 - tt1
    
    s1, c1 = jnp.sin(tt1), jnp.cos(tt1)
    s2, c2 = jnp.sin(tt2), jnp.cos(tt2)
    s21, c21 = jnp.sin(tt21), jnp.cos(tt21)
    M = jnp.array([[j1, c21], [c21, j2]])
    
    dd = d*jnp.array([2*w1 - w2, w2 - w1])
    #D = jnp.array([[2*d, -d], [-d, d]])
    
    b = s21*jnp.array([-w2**2, w1**2])
    tg = g/l1*jnp.array([mu1*s1, s2])
    
    p = jnp.dot(B, u) - b - tg - dd
    a = jnp.linalg.solve(M, p)
    dx = jnp.array([w1, w2, a[0], a[1]])
    
    # kinematics
    x1 = l1*jnp.array([s1, -c1])
    x2 = x1 + l2*jnp.array([s2, -c2])
    
    return dx, jnp.concatenate((x, x1, x2))
    

@jit
def butterfly_dynamics(params, x, u):
    '''
    Dynamics to simulate butterfly robot
    '''
    l1, l2, l3, J, d, B = params
    j12, j13, j23 = J[0,1], J[0,2], J[1,2]
    tt1, tt2, tt3, w1, w2, w3 = x
    w = x[3:]
    tt21 = tt2 - tt1
    tt31 = tt3 - tt1
    tt32 = tt3 - tt2
    
    s1, c1 = jnp.sin(tt1), jnp.cos(tt1)
    s2, c2 = jnp.sin(tt2), jnp.cos(tt2)
    s3, c3 = jnp.sin(tt3), jnp.cos(tt3)
    s21, c21 = jnp.sin(tt21), jnp.cos(tt21)
    s31, c31 = jnp.sin(tt31), jnp.cos(tt31)
    s32, c32 = jnp.sin(tt32), jnp.cos(tt32)
    
    M = J*jnp.array([
        [1, l1*c21, l1*c31], 
        [l1*c21, 1, l2*c32], 
        [l1*c31, l2*c32, 1]
        ])
    
    b = jnp.array([
        - j12*l1*s21*w2**2 - j13*l1*s31*w3**2, 
        j12*l1*s21*w1**2 - j13*l2*s32*w3**2,
        j13*(l1*s31*w1**2 + l2*s32*w2**2)
        ])
    tg = g*jnp.array([
        s1, 
        j12*s2,
        j13*s3
        ])
    
    dd = d*w

    p = jnp.dot(B, u) - b - tg - dd
    a = jnp.linalg.solve(M, p)
    dx = jnp.concatenate((w, a))
    
    # kinematics
    x1 = l1*jnp.array([s1, -c1])
    x2 = x1 + l2*jnp.array([s2, -c2])
    x3 = x2 + l3*jnp.array([s3, -c3])
    
    return dx, jnp.concatenate((x, x1, x2, x3))