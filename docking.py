"""
Spacecraft docking to a spinning space station [1]. Solution using lossless
convexification.

[1] https://www.youtube.com/watch?v=c4tPQYNpW9k

D. Malyuta -- ACL, University of Washington
B. Acikmese -- ACL, University of Washington

Copyright 2019 University of Washington. All rights reserved.
"""

import pickle
import numpy as np
import numpy.linalg as la
from numpy.linalg import matrix_power as mpow
import cvxpy as cvx

import lcvx
import tools

class Docker(lcvx.Problem,object):
    def __init__(self):
        super(Docker, self).__init__()
        cvx_opts = dict(solver=cvx.ECOS,verbose=False,abstol=1e-8,max_iters=100)
        
        # Physical parameters
        self.omega = np.array([0.*2*np.pi/60.,0.*2*np.pi/60.,1.*2*np.pi/60.]) # [rad/s] Space station spin
        self.rho1 = 1e-3#0.5e-2 # [m/s^2] Smallest control acceleration
        self.rho2 = 1e-2#0.5e-1 # [m/s^2] Largest control acceleration
        
        # Boundary conditions
        r0 = np.array([5.,5.,1e2])
        v0 = np.array([0.,0.,0.]) #-np.cross(self.omega,r0)
        rf = np.array([0.,0.,0.])
        vf = np.array([0.,0.,-1e-2])
        
        # RCS layout
        cone_angle = 0. # [deg] Thruster cone angle
        theta = phi = 30 # [deg] Basic pitch, roll
        cone_parameters = [dict(alpha=cone_angle,roll=phi,pitch=theta,yaw=0),
                           dict(alpha=cone_angle,roll=-phi,pitch=theta,yaw=0),
                           dict(alpha=cone_angle,roll=-phi,pitch=-theta,yaw=0),
                           dict(alpha=cone_angle,roll=phi,pitch=-theta,yaw=0),
                           dict(alpha=cone_angle,roll=phi+(180-2*phi),pitch=-theta,yaw=0),
                           dict(alpha=cone_angle,roll=-phi-(180-2*phi),pitch=-theta,yaw=0),
                           dict(alpha=cone_angle,roll=-phi-(180-2*phi),pitch=theta,yaw=0),
                           dict(alpha=cone_angle,roll=phi+(180-2*phi),pitch=theta,yaw=0),
                           dict(alpha=cone_angle,roll=90,pitch=0,yaw=0),
                           dict(alpha=cone_angle,roll=-90,pitch=0,yaw=0),
                           dict(alpha=cone_angle,roll=0,pitch=90,yaw=0),
                           dict(alpha=cone_angle,roll=0,pitch=-90,yaw=0)]
        self.C = [tools.make_cone(**param) for param in cone_parameters]
        self.Cn = [tools.make_cone(normal=True,**param) for param in cone_parameters]
        self.M = len(self.C) # Number of thrusters
        self.K = 4 # How many thrusters can be simultaneously active
        
        # Setup dynamical system
        S = lambda w: np.array([[0,-w[2],w[1]],[w[2],0,-w[0]],[-w[1],w[0],0]])
        Ac = np.block([[np.zeros((3,3)),np.eye(3)],[-mpow(S(self.omega),2),-2*S(self.omega)]])
        Bc = np.row_stack([np.zeros((3,3)),np.eye(3)])
        self.A,self.B = Ac,Bc
        nx,nu = Ac.shape[1], Bc.shape[1]
        A = cvx.Parameter(nx,nx)
        B = cvx.Parameter(nx,nu)
        
        # Scaling
        Dx = np.concatenate(np.abs([r0,v0]))
        Dx[Dx==0] = 1
        Dx = np.diag(Dx)
        Du = np.diag([self.rho2 for _ in range(nu)])
        
        # Optimization problem common parts
        self.N = 300 # Temporal solution
        x = [cvx.Parameter(nx)]+[Dx*cvx.Variable(nx) for _ in range(1,self.N+1)]
        xi = [cvx.Parameter()]+[cvx.Variable() for _ in range(1,self.N+1)]
        u = [[Du*cvx.Variable(nu) for __ in range(self.N)] for _ in range(self.M)]
        sigma = [cvx.Variable(self.N) for _ in range(self.M)]
        gamma = [cvx.Variable(self.N) for _ in range(self.M)]
        dt = cvx.Parameter()
        J2 = cvx.Parameter()
        
        self.zeta = 1
        cost_p2 = dt*self.N/100.+xi[-1]
        cost_p3 = dt*self.N+0.01*xi[-1] # but do golden search with dt*self.N
        
        self.constraints = []
        self.dual2idx = dict()
        def add_constraint(new_constraints,dual):
            """
            Add constraint(s).
            
            Parameters
            ----------
            new_constraints : list
                List of constraints to add.
            dual : str
                Dual variable name.
            """
            idx_start = len(self.constraints)
            self.constraints += new_constraints
            idx_end = len(self.constraints)
            if dual not in self.dual2idx:
                self.dual2idx[dual] = range(idx_start,idx_end)
            else:
                self.dual2idx[dual] += range(idx_start,idx_end)
        
        add_constraint([x[k+1] == A*x[k]+B*sum([u[i][k] for i in range(self.M)]) for k in range(self.N)],'nu_x')
        add_constraint([xi[k+1] == xi[k]+dt*sum([sigma[i][k] for i in range(self.M)]) for k in range(self.N)],'nu_xi')
        x[0].value = np.concatenate([r0,v0])
        xi[0].value = 0
        add_constraint([x[-1] == np.concatenate([rf,vf])],'nu_xN')
        for i in range(self.M):
            add_constraint([cvx.norm(u[i][k]) <= sigma[i][k] for k in range(self.N)],'lambda_sigma')
            add_constraint([gamma[i][k]*self.rho1 <= sigma[i][k] for k in range(self.N)],'lambda_rho1')
            add_constraint([sigma[i][k] <= gamma[i][k]*self.rho2 for k in range(self.N)],'lambda_rho2')
            add_constraint([gamma[i][k] >= 0 for k in range(self.N)],'lambda_gamma_low')
            add_constraint([gamma[i][k] <= 1 for k in range(self.N)],'lambda_gamma_high')
            add_constraint([self.C[i]*u[i][k] <= 0 for k in range(self.N)],'lambda_u')
        add_constraint([sum([gamma[i][k] for i in range(self.M)]) <= self.K for k in range(self.N)],'lambda_sum_gamma')
        
        # Problem 2 oracle
        def extract_variables():
            # Primal variables
            primal = dict()
            primal['x'] = np.column_stack([tools.cvx2arr(x[k]) for k in range(self.N+1)])
            primal['u'] = [np.column_stack([tools.cvx2arr(u[i][k]) for k in range(self.N)]) for i in range(self.M)]
            primal['sigma'] = [np.array([sigma[i][k].value for k in range(self.N)]) for i in range(self.M)]
            primal['gamma'] = [np.array([gamma[i][k].value for k in range(self.N)]) for i in range(self.M)]
            # Dual variables
            dual = dict()
            for name in cooper.dual2idx.keys():
                dual[name] = np.column_stack([tools.cvx2arr(self.constraints[k],dual=True) for k in self.dual2idx[name]])
            # Other (derived) variables
            misc = dict()
            misc['y'] = np.row_stack([self.Bd.T.dot(dual['nu_x'][:,k]) for k in range(self.N)])
            return primal,dual,misc
        
        p2 = cvx.Problem(cvx.Minimize(cost_p2),self.constraints)
        
        def problem2(tf):
            dt.value = tf/float(self.N)
            A.value,B.value = tools.discretize(Ac,Bc,dt.value)
            self.Ad = np.array(A.value)
            self.Bd = np.array(B.value)
            
            t = np.array([k*dt.value for k in range(self.N+1)])
            
            try:
                J = p2.solve(**cvx_opts)
                if p2.status=='infeasible':
                    return p2.status,np.inf,t,None,None,None
                else:
                    # All good, return solution
                    primal,dual,misc = extract_variables()
                    return p2.status,J,t,primal,dual,misc
            except cvx.SolverError:
                return 'cvx.SolverError',np.inf,t,None,None,None
            
        self.problem2 = lambda tf: problem2(tf)
        
        # Problem 3 oracle
        p3 = cvx.Problem(cvx.Minimize(cost_p3),self.constraints+[cost_p2<=J2])
        
        def problem3(tf,J):
            dt.value = tf/float(self.N)
            J2.value = J
            
            A.value,B.value = tools.discretize(Ac,Bc,dt.value)
            self.Ad = np.array(A.value)
            self.Bd = np.array(B.value)
            
            t = np.array([k*dt.value for k in range(self.N+1)])
            
            try:
                J = p3.solve(**cvx_opts)
                if p3.status=='infeasible':
                    return p3.status,np.inf,t,None,None,None
                else:
                    # All good, return solution
                    primal,dual,misc = extract_variables()
                    cost_mintime = tf
                    return p3.status,cost_mintime,t,primal,dual,misc
            except cvx.SolverError:
                return 'cvx.SolverError',np.inf,t,None,None,None
            
        self.problem3 = lambda tf,J2: problem3(tf,J2)

cooper = Docker()
J,t,primal,dual,misc = lcvx.solve(cooper,[0.,1000.],opt_tol=1e-4)
Jt = np.linspace(150,350,20)
J = tools.cost_profile(oracle = lambda tf: cooper.problem2(tf)[1],t_range=Jt)

##%% Post-process data

# Compute state and input in the inertial frame
nrot = cooper.omega/la.norm(cooper.omega)
w = la.norm(cooper.omega)
nx = np.array([[0,-nrot[2],nrot[1]],[nrot[2],0,-nrot[0]],[-nrot[1],nrot[0],0]])
nnT = np.outer(nrot,nrot)
R = lambda t: (np.cos(w*t)*np.eye(3)+np.sin(w*t)*nx+(1-np.cos(w*t))*nnT)
primal['x_inertial'] = np.row_stack([np.column_stack([R(t[k]).dot(primal['x'][:3,k]) for k in range(cooper.N+1)]),
                                     np.column_stack([R(t[k]).dot(primal['x'][3:,k]) for k in range(cooper.N+1)])])
primal['u_inertial'] = [np.column_stack([R(t[k]).dot(primal['u'][i][:,k]) for k in range(cooper.N)]) for i in range(cooper.M)]

pickle.dump(dict(rho1=cooper.rho1,rho2=cooper.rho2,N=cooper.N,M=cooper.M,
                 A=cooper.Ad,B=cooper.Bd,J=J,Jt=Jt,
                 K=cooper.K,C=cooper.C,t=t,primal=primal,dual=dual,misc=misc),open('solution.pkl','wb'))