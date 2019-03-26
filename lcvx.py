"""
Automatica '19 [1] Algorithm 1 and its directly related functions.

[1] https://arxiv.org/abs/1902.02726

D. Malyuta -- ACL, University of Washington
B. Acikmese -- ACL, University of Washington

Copyright 2019 University of Washington. All rights reserved.
"""

from tools import golden

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
    """
    # Solve phase 1 problem
    f = lambda tf: problem.problem2(tf)[:2]
    tf = golden(f,tf_range[0],tf_range[1],opt_tol)
    status,J,t,primal,dual,misc = problem.problem2(tf)
    if problem.zeta==1:
        f = lambda tf: problem.problem3(tf,J)[:2]
        tf = golden(f,tf_range[0],tf,opt_tol)
        status,J,t,primal,dual,misc = problem.problem3(tf,J)
    
    return J,t,primal,dual,misc
