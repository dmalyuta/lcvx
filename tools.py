"""
Helper functions.

D. Malyuta -- ACL, University of Washington
B. Acikmese -- ACL, University of Washington

Copyright 2019 University of Washington. All rights reserved.
"""

import sys
import numpy as np
import numpy.linalg as la
import scipy.linalg as sla
import cvxpy as cvx
import progressbar as progressbar

def golden(f,lb,ub,tol,name=None):
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
    name: str, optional
        Display a name for the progressbar.
    
    Returns
    -------
    x : float
        Minimum location, \in [lb,ub].
    solver_time : float
        Sum of solver times for all calls to the optimizer
    """
    # Maintains the interval [x1,(x2,x4),x3] where [x1,x3] brackets the
    # minimum and x2, x4 are intermediate points used to update the bracket
    solver_time = 0
    x = [lb,np.nan,ub]
    fx2 = np.nan
    phi = (1+np.sqrt(5))/2. # Golden ratio
    icount = np.ceil(np.log((ub-lb)/(2*tol))/np.log(phi)) if ub>lb else 0
    if icount == 0:
        x[1] = ub
    name = '' if name is None else name
    widgets=[name,'[%.2f,%.2f,%.2f]'%(x[0],np.nan,x[2]),' ',progressbar.Bar(),' (',
             progressbar.ETA(), ') (solver time: ','0.00',' s)']
    for i in progressbar.progressbar(range(int(icount)),widgets=widgets):
        try:
            if np.isnan(x[1]):
                x[1] = (phi*x[0]+x[2])/(phi+1.)
                _,fx2,time = f(x[1])
                solver_time += time
                widgets[-2] = '%.2e'%(solver_time)
            x4 = x[0]+x[2]-x[1]
            status,fx4,time = f(x4)
            solver_time += time
            if fx4<=fx2:
                x[0] = x[1]
                x[1] = x4
                fx2 = fx4
            else:
                x[2] = x[0]
                x[0] = x4
            y = np.sort(x)
            widgets[1] = '[%.2f,%.2f,%.2f] {%s}'%(y[0],y[1],y[2],status)
            widgets[-2] = '%.2e'%(solver_time)
        except KeyboardInterrupt:
            sys.exit()
    
    # Get the location that is feasible, starting from the upper bound
    y = np.sort(x)
    for i in range(2,-1,-1):
        status,_,time = f(y[i])
        solver_time += time
        if status=='optimal' or status=='optimal_inaccurate':
            x = y[i]
            break
    widgets[-2] = '%.2e'%(solver_time)
    
    return x,solver_time

def cost_profile(oracle,t_range):
    """
    Get the cost function profile.
    
    Parameters
    ----------
    oracle : callable
        Call signature cost=oracle(time) where time (float) is the final time
        and cost (float) is the optimal cost.
    t_range : list
        List of times to compute the cost at.
        
    Returns
    -------
    J : array
        Array of optimal cost values at those times.
    """
    J = np.array([oracle(t) for t in progressbar.progressbar(t_range)])
    return J

def discretize(Ac,Bc,dt):
    """Dynamics discretization"""
    M = sla.expm(np.block([[Ac,Bc],
                           [np.zeros([Bc.shape[1],
                                      Ac.shape[1]+Bc.shape[1]])]])*dt)
    A = M[:Ac.shape[0],:Ac.shape[0]]
    B = M[:Ac.shape[0],Ac.shape[0]:]
    return A,B

def cvx2arr(x,dual=False):
    """Convert CVX variable to an array"""
    return np.array(x.value.T if not dual else x.dual_value.T).flatten()

def project(y,C,abstol=1e-7):
    """
    Project vector y onto the polytopic set {u: C*u<=0}.
    
    Parameters
    ----------
    y : array
        Vector to project.
    C : array
        Matrix whose rows are the polytopic set facet normals.
    abstol : float, optional
        Absolute tolerance on duality gap.
        
    Returns
    -------
    z : array
        Projection of y.
    """
    z = cvx.Variable(y.size)
    cost = cvx.Minimize(cvx.norm2(y-z))
    constraints = [C*z <= 0]
    problem = cvx.Problem(cost,constraints)
    problem.solve(solver=cvx.ECOS,verbose=False,abstol=abstol,reltol=np.finfo(np.float64).eps)
    if problem.status!='optimal':# and problem.status!='optimal_inaccurate':
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
                           Ry(-angle-alpha/2.).dot(nhat_base),
                           -nhat_base])
    R = Rz(yaw).dot(Ry(pitch)).dot(Rx(roll)) # Overall active rotation
    C = C_base.dot(R.T)
    return C

def Dsvd(A):
    """
    SVD wrapper that accepts empty matrices.
    
    Parameters
    ----------
    A : array
        Matrix whose SVD to calculate.
        
    Returns
    -------
    u,s,vh : array
        Same as outputs of MATLAB's svd.
    """
    if A.shape[0]==0 or A.shape[1]==0:
        u = np.empty((0,0)) if A.shape[0]==0 else np.eye(A.shape[0])
        s = np.empty(A.shape)
        v = np.empty((0,0)) if A.shape[1]==0 else np.eye(A.shape[1])
    else:
        u,s,vh = la.svd(A)
        s = np.diag(s)
        v = vh.T
    return u,s,v

def Drank(A,Leps=None):
    """
    Matrix rank that accepts empty matrices.
    
    Parameters
    ----------
    A : array
        Matrix whose rank to compute.
    Leps : float, optional
        Tolerance of calculation.
        
    Returns
    -------
    : int
        Matrix rank.
    """
    if A.shape[0]==0 or A.shape[1]==0:
        return 0
    else:
        return la.matrix_rank(A,Leps) if Leps is not None else la.matrix_rank(A)
    
def Dnull(A):
    """
    Nullspace computation that accepts empty matrices.
    
    Parameters
    ----------
    A : array
        Matrix whose nullspace to compute.
        
    Returns
    -------
    : array
        Matrix nullspace.
    """
    if A.shape[0]==0 or A.shape[1]==0:
        return np.eye(A.shape[1])
    else:
        return sla.null_space(A)

def Brank(A,Leps=None):
    """
    r = Brank(A,Leps)
    --------------------
    Behcet Acikmese
    Danylo Malyuta
    
    1. June, 2007
    2. April, 2019 (Python version)
    
    Parameters
    ----------
    A : array
        Basis matrix whose rank to compute.
    Leps : float, optional
        Tolerance of calculation.
        
    Returns
    -------
    r : int
        Rank of A.
    """
    r = Drank(A,Leps)
    nA = Dnull(A)
    nAt = Dnull(A.T)
    nr = nA.shape[1]
    nrt = nAt.shape[1]
    r2 = A.shape[1]-nr
    r3 = A.shape[0]-nrt
    r3 = np.min([r2,r3,Drank(A.T.dot(A))]) #strong rank:)
    r = np.min([r,r3])
    return r

def Bim(A,Leps=None):
    """
    W = Bim(A,Leps)
    --------------------
    im(V) = im(A) and V is onto s.t. V'V =I
    --------------------
    Behcet Acikmese
    Danylo Malyuta
    
    1. June, 2007
    2. April, 2019 (Python version)
    
    Parameters
    ----------
    A : array
        Basis matrix which to reduce to minimal basis.
    Leps : float, optional
        Tolerance of calculation.
        
    Returns
    -------
    W : array
        Onto version of A.
    Sv : array
        Singular values.
    """
    U,S = Dsvd(A)[:2]
    Sd = np.diag(S)
    r = Brank(A,Leps)
    n = U.shape[0]
    if r>0:
        W = U[:,:r]
        Sv = Sd[:r]
        if r==n:
            W = np.eye(r)
    else:
        W = np.zeros((n,0))
        Sv = np.zeros((1,0))
    return W, Sv   

def BsumS(A,B,Leps=None):
    """
    W = BsumS(A,B,Leps)
    --------------------
    im(V) = sum of subspaces im(A) and im(B)
    --------------------
    Behcet Acikmese
    Danylo Malyuta
    
    1. September 2006
    2. June, 2007 (update)
    3. April, 2019 (Python version)
    
    Parameters
    ----------
    A : array
        Basis matrix of subspace A.
    B : array
        Basis matrix of subspace B.
    Leps : float, optional
        Tolerance for rank calculation.
        
    Returns
    -------
    W : array
        Basis matrix of im(A)+im(B).
    Sw : array
        Singular values.
    """
    n = A.shape[0]
    m = B.shape[0]
    if n != m:
        print 'Dimensions of subspaces do NOT match'
        W = np.empty(0)
        Sv = np.empty(0)
    else:
        Ua,ra = Dsvd(A)[0],Brank(A)
        Ub,rb = Dsvd(B)[0],Brank(B)
        T = np.column_stack([Ua[:,:ra],Ub[:,:rb]])
        W,Sv = Bim(T,Leps)
    return W, Sv

def Bker(A,Leps=None):
    """
    W = Bker(A,Leps)
    --------------------
    im(V) = ker(A) and V is onto s.t. V'V =I
    --------------------
    Behcet Acikmese
    Danylo Malyuta
    
    1. June, 2007
    2. April, 2019 (Python version)
    
    Parameters
    ----------
    A : array
        Basis matrix whose kernel to compute.
    Leps : float, optional
        Tolerance of calculation.
        
    Returns
    -------
    W : int
        Basis matrix of kernel of A.
    Sv : array
        Singular values.
    """
    S,V = Dsvd(A)[1:]
    Sd = np.diag(S)
    V = V.T
    r = Brank(A,Leps)
    Sv = Sd[:r]
    n = V.shape[0]
    if r==n:
        W = np.zeros((n,0))
    else:
        W = V[r:,:].T
    return W,Sv

def BVinv(A,V,Leps=None):
    """
    W = BVinv(A,V,Leps)
    --------------------
    Finds the A^-1(imV): {x: Ax \in imV}
    Leps optional for rank computation
    Uses: A^-1(imV) = ker(A' kerV')'
    --------------------
    Behcet Acikmese
    
    1. June, 2007
    2. April, 2019 (Python version)
    
    Parameters
    ----------
    A : array
        Matrix whose inverse set map to compute.
    V : array
        Basis matrix of set.
    Leps : float, optional
        Tolerance of calculation.
        
    Returns
    -------
    W : array
        Basis matrix of A^-1(imV).
    """
    n = A.shape[0]
    nc = A.shape[1]
    m = V.shape[0]
    
    if n!=m:
        print 'Error: dimensions mismatch'
        W = np.zeros((nc,0))
    else:
        kV = Bker(V.T)[0]
        AkV = A.T.dot(kV)
        W = Bker(AkV.T)[0]
    return W

def weak_unobsv_sub(A,B,C,D):
    """
    Find the weakly observable subspace for the linear system
    \Sigma = (A,B,C,D).
    Based on Trentelman et al. page 162
    
    Matt Harris, April 16, 2013
    Danylo Malyuta, April 6, 2019 (Python version)
    
    Parameters
    ----------
    A,B,C,D : array
        Matrices of commensurate dimensions defining the dynamica system.
    
    Returns
    -------
    V : array
        Basis matrix of the weakly unobservable subspace.
    """
    n = A.shape[0]
    m = B.shape[1]
    p = C.shape[0]
    
    AC = np.row_stack([A,C])
    BD = np.row_stack([B,D])
    
    V = np.eye(n)
    
    for i in range(n):
        q = V.shape[1]
        S1 = np.block([[V,np.zeros((n,m))],[np.zeros((p,q+m))]])
        S2 = BsumS(S1,BD)[0]
        V = BVinv(AC,S2)
    
    return V