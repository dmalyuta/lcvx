"""
Helper functions.

D. Malyuta -- ACL, University of Washington
B. Acikmese -- ACL, University of Washington

Copyright 2019 University of Washington. All rights reserved.
"""

import numpy as np
import numpy.linalg as la
import scipy.linalg as sla
import cvxpy as cvx
import progressbar as progressbar

def golden(f,lb,ub,tol):
    """
    Golden search for a minimum.
    
    Parameters
    ----------
    f : function
        Oracle which returns status (e.g. of the optimization problem) and
        f(x). Must be unimodal \in [lb,ub].
    lb : float
        Search interval lower bound.
    ub : float
        Search interval upper bound.
    tol : float
        Convergence tolerance, such that the minimum is within tol units away
        from the returned x.
    
    Returns
    -------
    x : float
        Minimum location, \in [lb,ub].
    fx : float
        Oracle value.
    """
    # Maintains the interval [x1,(x2,x4),x3] where [x1,x3] brackets the
    # minimum and x2, x4 are intermediate points used to update the bracket
    x = [lb,np.nan,ub]
    fx2 = np.nan
    phi = (1+np.sqrt(5))/2. # Golden ratio
    icount = np.ceil(np.log((ub-lb)/(2*tol))/np.log(phi)) if ub>lb else 0
    if icount == 0:
        x[1] = ub
    widgets=['[%.2f,%.2f,%.2f]'%(x[0],np.nan,x[2]),' ',progressbar.Bar(),' (', progressbar.ETA(), ') ']
    for i in progressbar.progressbar(range(int(icount)),widgets=widgets):
        if np.isnan(x[1]):
            x[1] = (phi*x[0]+x[2])/(phi+1.)
        x4 = x[0]+x[2]-x[1]
        status,fx4 = f(x4)
        if np.isnan(fx2):
            _,fx2 = f(x[1])
        if fx4<=fx2:
            x[0] = x[1]
            x[1] = x4
            fx2 = fx4
        else:
            x[2] = x[0]
            x[0] = x4
        y = np.sort(x)
        widgets[0] = '[%.2f,%.2f,%.2f] {%s}'%(y[0],y[1],y[2],status)
    
    x = x[1]#x[2] if x[0]<x[2] else x[0]
    fx = f(x)
    
    return x,fx

def discretize(Ac,Bc,dt):
    """Dynamics discretization"""
    M = sla.expm(np.block([[Ac,Bc],
                           [np.zeros([Bc.shape[1],
                                      Ac.shape[1]+Bc.shape[1]])]])*dt)
    A = M[:Ac.shape[0],:Ac.shape[0]]
    B = M[:Ac.shape[0],Ac.shape[0]:]
    return A,B

def cvx2arr(x):
    """Convert CVX variable to an array"""
    return np.array(x.value.T).flatten()

def project(y,C):
    """
    Project vector y onto the polytopic set {u: C*u<=0}.
    
    Parameters
    ----------
    y : array
        Vector to project.
    C : array
        Matrix whose rows are the polytopic set facet normals.
        
    Returns
    -------
    z : array
        Projection of y.
    """
    z = cvx.Variable(y.size)
    cost = cvx.Minimize(cvx.norm2(y-z))
    constraints = [C*z <= 0]
    problem = cvx.Problem(cost,constraints)
    problem.solve(solver=cvx.ECOS,verbose=False)
    if problem.status!='optimal' and problem.status!='optimal_inaccurate':
        raise RuntimeError('Projection operation failed')
    return cvx2arr(z)

def pbh(A,B):
    """
    Popov-Belevitch-Hautus controllability test for the pair {A,B}.
    
    Returns
    ----------
    : bool
        ``True`` if the pair {A,B} is controllable.
    """
    nx = A.shape[1]
    eigvals,eigvecs = la.eig(A)
    for eigval,eigvec in zip(eigvals,eigvecs):
        pbh_mat = np.column_stack([eigval*np.eye(A.shape[1])-A,B])
        if la.matrix_rank(pbh_mat)<nx:
            return False
    return True

def make_cone(alpha,roll,pitch,yaw,normal=False):
    """
    Generates a four-sided cone {u: C*u<=0} with opening angle alpha and
    pointed according to Tait-Bryan convention. The cone is rotated starting
    from a +z orientation.
    
    Parameters
    ----------
    alpha : float
        Cone opening angle (angle between two opposing hyperplanes) in degrees.
    roll : float
        Roll angle about x'' in degrees.
    pitch : float
        Pitch angle aboubt y' in degrees.
    yaw : float
        Yaw angle about z in degrees.
    normal : float
        Compute instead the normal cone to {u: C*u<=0}, i.e. the set
        {v: N*v<=0} such that v^T*u<=0 for all u such that C*u<=0.
    
    Returns
    -------
    C : array
        Matrix whose rows are the facet outwarding-facing normals (of the cone
        or of the normal cone).
    """
    alpha = np.deg2rad(alpha)
    roll = np.deg2rad(roll)
    pitch = np.deg2rad(pitch)
    yaw = np.deg2rad(yaw)
    c = lambda u: np.cos(u)
    s = lambda u: np.sin(u)
    Rx = lambda u: np.array([[1,0,0],[0,c(u),-s(u)],[0,s(u),c(u)]])
    Ry = lambda u: np.array([[c(u),0,s(u)],[0,1,0],[-s(u),0,c(u)]])
    Rz = lambda u: np.array([[c(u),-s(u),0],[s(u),c(u),0],[0,0,1]])
    # Compute the non-rotated cone
    angle = 0. if normal else np.pi/2.
    nhat_base = np.array([0,0,1])
    C_base = np.row_stack([Rx(angle+alpha/2.).dot(nhat_base),
                           Rx(-angle-alpha/2.).dot(nhat_base),
                           Ry(angle+alpha/2.).dot(nhat_base),
                           Ry(-angle-alpha/2.).dot(nhat_base)])
    R = Rz(yaw).dot(Ry(pitch)).dot(Rx(roll)) # Overall active rotation
    C = C_base.dot(R.T)
    return C