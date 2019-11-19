"""
Automatica '19 [1] Algorithm 1 and its directly related functions.

[1] https://arxiv.org/abs/1902.02726

D. Malyuta -- ACL, University of Washington
B. Acikmese -- ACL, University of Washington

Copyright 2019 University of Washington. All rights reserved.
"""

import tools

class Problem:
    def solve(self,tf):
        """
        Relaxed problem.
        
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

def solve(problem,tf_range,opt_tol=1e-2):
    """
    Lossless convexification algorithm.
    
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
        status,J,t,primal,dual,misc,solver_time = problem.solve(tf)
        return status,J,solver_time
    tf,golden_time = tools.golden(f,tf_range[0],tf_range[1],opt_tol,'Problem R ')
    status,J,t,primal,dual,misc,solver_time = problem.solve(tf)
    total_solver_time = golden_time+solver_time
    
    return J,t,primal,dual,misc,total_solver_time
