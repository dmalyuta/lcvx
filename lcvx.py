"""
Automatica '19 [1] Algorithm 1 and its directly related functions.

[1] https://arxiv.org/abs/1902.02726

D. Malyuta -- ACL, University of Washington
B. Acikmese -- ACL, University of Washington

Copyright 2019 University of Washington. All rights reserved.
"""

import numpy as np
import numpy.linalg as la
import cvxpy as cvx

import tools

class Problem:
    def __init__(self):
        self.zeta = 1 # Set this to the value of zeta in Problem 1
    
    def problem2(self,tf):
        """
        Problem 2 - the original relaxation.
        
        Parameters
        ----------
        tf : float
            Terminal time.
            
        Returns
        -------
        status : string
            Optimizer status.
        J : float
            The optimal cost.
        t : array
            The time grid.
        primal : dict
            Primal variables.
        dual : dict
            Dual variables.
        misc : dict
            Other variables.
        solver_time : float
            Optimizer time.
        """
        raise NotImplementedError('problem2 not implemeted!')
        
    def problem3(self,tf,J2):
        """
        Problem 3 - companion minimum-time problem. Same inputs/outputs as
        problem2 apart from J2.
        
        Parameters
        ----------
        J2 : float
            Optimal cost of Problem 2.
        """
        raise NotImplementedError('problem3 not implemeted!')

def solve(problem,tf_range,opt_tol=1e-2):
    """
    Automatica '19 Algorithm 1.
    
    Parameters
    ----------
    problem : Problem
        The problem that is to be solved.
    tf_range : list
        Optimal terminal time search interval.
    opt_tol : float, optional
        Optimality tolerance for golden search in terms of how tight a bound to
        place on the true optimal final time.
    
    Returns
    -------
    [J,t,primal,dual,misc] : [float,array,dict,dict,dict]
        Same meaning as the output of problem.problem2 and problem.problem3.
    total_solver_time : float
        The sum of solver times over all calls to the optimizer.
    """
    # Solve phase 1 problem
    def f(tf):
        status,J,t,primal,dual,misc,solver_time = problem.problem2(tf)
        return status,J,solver_time
    tf,golden_time = tools.golden(f,tf_range[0],tf_range[1],opt_tol,'Problem 2 ')
    status,J,t,primal,dual,misc,solver_time = problem.problem2(tf)
    total_solver_time = golden_time+solver_time
    if problem.zeta==1:
        def f(tf,J2):
            status,J,t,primal,dual,misc,solver_time = problem.problem3(tf,J2)
            return status,J,solver_time
        f3 = lambda tf: f(tf,J)
        tf3,golden_time = tools.golden(f3,tf_range[0],tf,opt_tol,'Problem 3 ')
        tf = tf3 if tf3 is not None else tf
        status,J,t,primal,dual,misc,solver_time = problem.problem3(tf,J)
        total_solver_time += golden_time+solver_time
    
    return J,t,primal,dual,misc,total_solver_time

def check_conditions_123(problem,big_M=1e3,dt=None,N_sim=None):
    """
    Check lossless convexification guarantee conditions 1-3.
    
    Parameters
    ----------
    problem : Problem
        Problem, which is expected to have the member variables:
            - A (array): zero-input state dynamics
            - B (array): input-to-state dynamics mapping
            - C (list): list of input pointing cones
            - K (int): number of inputs that can be simultaneously active
            - M (int): total number of inputs
    big_M : float, optional
        Big-M coefficient for mixed-integer problem formulation of some of the
        condition checks.
    dt : float, optional
        Discretization time step.
    N_sim : int, optional
        Number of time steps to simulate for Condition 2 and 3 MILPs.
    
    Returns
    -------
    : bool
        ``True`` if all the conditions hold.
    info : dict
        Which condition failed and some related information.
    """
    cvxopts = dict(solver=cvx.GUROBI,verbose=False)
    # Parameters
    A,B,C,Dx,Du,K,M = problem.A,problem.B,problem.C,problem.Dx,problem.Du,problem.K,problem.M
    A = la.inv(Dx).dot(A).dot(Dx) # Scaled state to unit box
    B = la.inv(Dx).dot(B).dot(Du) # Scaled input to unit box
    nx = A.shape[1]
    #
    # Condition 1: {A,B} controllable
    #
    if not tools.pbh(A,B):
        print "[Condition 1] system is uncontrollable"
        info = dict(condition=1)
        return False, info
    #
    # Condition 2: if y can evolve on normal cone boundary, it projects onto K better alternatives
    #
    # Simulation length
    min_wn = np.min([eignorm for eignorm in np.abs(la.eigvals(A)) if eignorm!=0])
    tau = 1/min_wn # Time constant
    dt = tau/10 if dt is None else dt # Sampling time step
    steps_per_time_constant = np.int32(tau/dt) # Number of time steps per time constant
    N_sim = 2*steps_per_time_constant if N_sim is None else N_sim # How many time steps to simulate
    print "Discretization: dt=%.2e s, N_sim=%d, T_sim=%.2e s"%(dt,N_sim,dt*(N_sim-1))
    # Optimizable problem variables
    N = [tools.normal_cone(C[i]) for i in range(M)]
    n_N = [N[i].shape[0] for i in range(M)]
    Ad = tools.discretize(-A.T,B,dt)[0]
    adj = [cvx.Variable(nx) for t in range(N_sim)]
    y = [B.T*adj[t] for t in range(N_sim)]
    z = [[cvx.Bool(n_N[k]) for k in range(M)] for t in range(N_sim)]
    eps = 1 # Strict inequality offset (exploit that constraints are homogeneous)
    # Check condition for all normal cones' facets
    cost = cvx.Minimize(0)
    for i in range(M): # Loop through normal cones
        for j in range(n_N[i]): # Loop through normal cone facets
            if not tools.pbh(A,B.dot(N[i][j])):
                print("[Condition 2] normal cone i=%d facet j=%d uncontrollable... "%(i,j)),
                # y can evolve on normal cone facet with normal N[i][j]
                # check that it always projects onto K better alternatives
                constraints = []
                # Dynamics
                constraints += [adj[t+1] == Ad*adj[t] for t in range(N_sim-1)]
                # Make sure adj[0] non-trivial
                b0 = cvx.Bool(C[i].shape[0])
                constraints += [eps-big_M*(1-b0[l]) <= C[i][l]*y[0] for l in range(C[i].shape[0])]
                constraints += [C[i][l]*y[0] <= big_M*b0[l] for l in range(C[i].shape[0])]
                constraints += [sum(b0) >= 1]
                # All other constraints
                for t in range(N_sim):
                    constraints += [N[i]*y[t] <= 0]
                    constraints += [N[i][j]*y[t] == 0]
                    constraints += [eps-big_M*(1-z[t][k][l]) <= N[k][l]*y[t] for k in range(M) if k!=i for l in range(n_N[k])]
                    constraints += [N[k][l]*y[t] <= big_M*z[t][k][l] for k in range(M) if k!=i for l in range(n_N[k])]
                    constraints += [sum([z[t][k][l] for k in range(M) if k!=i for l in range(n_N[k])]) <= K-1]
                problem = cvx.Problem(cost,constraints)
                try:
                    problem.solve(**cvxopts)
                    # If got here, a feasible solution was found which means
                    # the condition failed
                    info = dict(condition=2,i=i,j=j,N_i=N[i],N_ij=N[i][j],
                                adj0=tools.cvx2arr(adj[0]))
                    print "<K better alternatives"
                    return False, info
                except cvx.SolverError:
                    print ">=K better alternatives OK"
                    pass
    #
    # Condition 3: if y can evolve on equiprojection manifold of two input
    # pointing sets, the two projections are unambiguously among or not among
    # the K best projections
    #
    # NB: assumes that the input pointing cones are ray-like!
    #
    nhat = [-C[i][-1]/la.norm(C[i][-1]) for i in range(M)]
    w = [cvx.Bool(M) for t in range(N_sim)]
    v = [cvx.Bool(M) for t in range(N_sim)]
    z = cvx.Bool(N_sim-1)
    # Check condition for all equiprojection manifolds
    for i in range(M): # Loop through all cones
        for j in range(M): # Loop through partner cones
            if i==j:
                continue # Skip same-cone pairs
            if not tools.pbh(A,B.dot(nhat[i]-nhat[j])):
                print("[Condition 3] pair (i=%d,j=%d) uncontrollable... "%(i,j)),
                # y can evolve on equiprojection manifold between the ray-like
                # cones i and j. Check that it is always unambiguously among
                # or not among the K best projections
                constraints = []
                # Dynamics
                constraints += [adj[t+1] == Ad*adj[t] for t in range(N_sim-1)]
                # Make sure adj[0] non-trivial
                constraints += [nhat[i]*y[0] >= eps]
                # All other constraints
                for t in range(N_sim):
                    constraints += [(nhat[i]-nhat[j])*y[t] == 0]
                    c = nhat[i]*y[t]
                    constraints += [c-big_M*(1-w[t][k]) <= nhat[k]*y[t] for k in range(M)]
                    constraints += [nhat[k]*y[t] <= c-eps+big_M*w[t][k] for k in range(M)]
                    constraints += [c+eps-big_M*v[t][k] <= nhat[k]*y[t] for k in range(M)]
                    constraints += [nhat[k]*y[t] <= c+big_M*(1-v[t][k]) for k in range(M)]
                    if t>0:
                        constraints += [sum([w[t][k] for k in range(M)]) >= (K+1)*z[t-1]]
                        constraints += [sum([v[t][k] for k in range(M)]) >= (M-K+1)*z[t-1]]
                    if t<N_sim-1:
                        constraints += [sum([w[t][k] for k in range(M)]) >= (K+1)*z[t]]
                        constraints += [sum([v[t][k] for k in range(M)]) >= (M-K+1)*z[t]]
                # Proxy for the condition being violated for non-trivial time
                constraints += [sum(z) >= 1]
                problem = cvx.Problem(cost,constraints)
                try:
                    problem.solve(**cvxopts)
                    # If got here, a feasible solution was found which means
                    # the condition failed
                    info = dict(condition=3,i=i,j=j,nhat_i=nhat[i],nhat_j=nhat[j],
                                equiproj_plane_normal=nhat[i]-nhat[j],
                                adj0=tools.cvx2arr(adj[0]),y0=tools.cvx2arr(y[0]),
                                z=tools.cvx2arr(z),
                                # Number of cones onto which y projects as much of more positively than onto cones i and j
                                # Should be >K at least twice
                                ge_proj_count=[sum([nhat[i].dot(tools.cvx2arr(y[t]))>=
                                                    nhat[0].dot(tools.cvx2arr(y[t]))
                                               for i in range(M)]) for t in range(N_sim)],
                                # Number of cones onto which y projects as much of less positively than onto cones i and j
                                # Should be >M-K at least twice at same locations where ge_proj_count is >K
                                le_proj_count=[sum([nhat[i].dot(tools.cvx2arr(y[t]))<=
                                                    nhat[0].dot(tools.cvx2arr(y[t]))
                                               for i in range(M)]) for t in range(N_sim)])
                    print "ambiguous"
                    return False, info
                except cvx.SolverError:
                    print "unambiguous OK"
                    pass
    info = dict()
    return True, info
