# -*- coding: utf-8 -*-
import numpy as np
from scipy.linalg import solve_discrete_are, solve_continuous_are, solve, eigvals
import jax.numpy as jnp
from jax import jit
from dynamics import g

def dlqr(A, B, q, rho=1.0):
    '''
    Discrete-time LQR with diagonal Q and scalar R
    '''
    nx, nu = B.shape
    
    Q = np.diag(q)
    R = rho*np.eye(nu)
    
    P = solve_discrete_are(A, B, Q, R)
    F = solve(R + B.T@P@B, B.T@P@A, sym_pos=True)
    
    return F, P, eigvals(A - B@F)

def clqr(A, B, q, rho=1.0):
    '''
    Continuous-time LQR with diagonal Q and scalar R
    '''
    nx, nu = B.shape
    
    Q = np.diag(q)
    R = rho*np.eye(nu)
    
    P = solve_continuous_are(A, B, Q, R)
    F = solve(R, B.T@P, sym_pos=True)
    
    return F, P, eigvals(A - B@F)

def clqr_full(A, B, Q, R=1.0):
    '''
    Continuous-time LQR with full Q and R
    '''
    nx, nu = B.shape
        
    P = solve_continuous_are(A, B, Q, R)
    F = solve(R, B.T@P, sym_pos=True)
    
    return F, P, eigvals(A - B@F)

def clqra(A, B, c, q, rho=1.0):
    '''
    Continuous-time LQR with diagonal Q and scalar R for an affine system
    '''
    nx, nu = B.shape
    
    Q = np.diag(q)
    R = rho*np.eye(nu)
    
    P = solve_continuous_are(A, B, Q, R)
    
    RiBt = solve(R, B.T, sym_pos=True)
    p = solve(A.T - P@B@RiBt, -P@c)
    
    F = RiBt@P # state gain matrix
    f = RiBt@p # constant input
    
    return F, f, P, p, eigvals(A - B@F)

#%% BEETLE CONTROL FUNCTIONS

f = np.array([[0, 1], [-1 ,0]])
@jit
def get_x1(x, l1, l2):
    '''
    Inverse kinematics to compute the two possible joint 1 positions for the 
    beetle robot given a joint 2 position
    '''
    d = jnp.sqrt(jnp.sum(x**2))
    l = (l1**2 - l2**2 + d**2)/2/d
    l = jnp.clip(l, -l1, l1)
    h = jnp.sqrt(l1**2 - l**2)
    
    center = x*l/d
    side = jnp.array([x[1], -x[0]])*h/d

    return jnp.stack((center + side, center - side), axis=0)

def to_joints(tt, l1, l2):
    '''
    From link angles to joint positions
    '''
    tt1, tt2 = tt
    s1, c1 = jnp.sin(tt1), jnp.cos(tt1)
    s2, c2 = jnp.sin(tt2), jnp.cos(tt2)

    x1 = l1*jnp.array([s1, -c1])
    x2 = x1 + l2*jnp.array([s2, -c2])
    
    return x1, x2

def to_angles(x1, x2):
    '''
    From joint positions to link angles
    '''
    x21 = x2 - x1
    tt1 = jnp.arctan2(x1[0], -x1[1])
    tt2 = jnp.arctan2(x21[0], -x21[1])
    return jnp.array([tt1, tt2])

def beetle_pd_controller(controller_params, x, x2ref, x1_prev):
    '''
    PD controller for beetle robot
    '''
    l1, l2, mu1, Kg, Kp, Kd, R = controller_params
    tt1, tt2, w1, w2 = x
    tt, w = x[:2], x[2:]

    s1, c1 = jnp.sin(tt1), jnp.cos(tt1)
    s2, c2 = jnp.sin(tt2), jnp.cos(tt2)

    x1, x2 = to_joints(x[:2], l1, l2)
    
    x1ref = get_x1(x2ref, l1, l2)
    x1refo = x1ref
    
    idx_current = jnp.argmin(jnp.sum((x1ref - x1)**2, axis=1), axis= 0)
    idx_previous = jnp.argmin(jnp.sum((x1ref - x1_prev)**2, axis=1), axis= 0)

    
    initialize = jnp.any(jnp.isnan(x1_prev))
    idx = jnp.where(initialize, idx_current, idx_previous)   
    x1ref = x1ref[idx]
    
    ttref = to_angles(x1ref, x2ref)
    tterr = jnp.mod(tt - ttref + np.pi, 2*np.pi) - np.pi
    
    tg = g/l1*jnp.array([mu1*s1, s2])
    
    ut = jnp.dot(Kg, tg - jnp.dot(Kd, w) - jnp.dot(Kp, tterr))
    return jnp.dot(R, ut), (ut, x2, ttref), x1ref
    
#%%

def butterfly_pd_controller(params, controller_params, x, x3ref, controller_state=None, maneuverability_power=2):
    '''
    PD controller for butterfly robot
    '''
    l1, l2, l3, j12, j13, Kg, Br = params
    Kp, Kd, Km = controller_params
    tt1, tt2, tt3, w1, w2, w3 = x
    tt, w = x[:3], x[3:]

    s1, c1 = jnp.sin(tt1), jnp.cos(tt1)
    s2, c2 = jnp.sin(tt2), jnp.cos(tt2)
    s3, c3 = jnp.sin(tt3), jnp.cos(tt3)

    x1 = l1*jnp.array([s1, -c1])
    x2 = x1 + l2*jnp.array([s2, -c2])
    x3 = x2 + l3*jnp.array([s3, -c3])

    # kinematics jacobian
    J = jnp.array([
        [l1*c1, l2*c2, l3*c3],
        [l1*s1, l2*s2, l3*s3],
        [0, 0, 1]
        ])
    v3 = jnp.dot(J[:2], w)

    # adds gradient of a small potential term to try to avoid singular 
    # configurations
    tt21 = tt2 - tt1
    tt32 = tt3 - tt2
    s21, c21 = jnp.sin(tt21), jnp.cos(tt21)
    s32, c32 = jnp.sin(tt32), jnp.cos(tt32)
    tm = Km*maneuverability_power*(c21*c32)**(maneuverability_power-1)*jnp.array([
        -c32*s21,
        c32*s21 - c21*s32,
        c21*s32
        ])
    
    # gravity comp
    tg = g*jnp.array([s1, j12*s2, j13*s3])
    # jacobian transpose PD
    tc = jnp.dot(J[:2].T, - Kd*v3 - Kp*(x3 - x3ref)) 
    # add gravity comp, control and "maneuverability gradient"
    u = jnp.dot(Br.T, jnp.dot(Kg, tg + tc + tm))
    return u, (u, x3, jnp.sin(tt2 - tt1)), None