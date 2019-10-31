"""
2D rocket landing with a thruster that can produce either low thrusts with a
large gimbal angle, or large thrusts with a small gimbal angle.

D. Malyuta -- ACL, University of Washington
B. Acikmese -- ACL, University of Washington

Copyright 2019 University of Washington. All rights reserved.
"""

import numpy as np
import numpy.linalg as la
import pickle
import cvxpy as cvx

import lcvx
import tools
import landing_plots

#%% Problem definition

class Lander(lcvx.Problem,object):
    def __init__(self,agl,micp=False):
        """
        Parameters
        ----------
        agl : float
            Initial altitude above ground level (AGL).
        micp : bool, optional
            Set to ``True`` to solve the problem via mixed-integer programming.
        """
        super(Lander, self).__init__()
        
        #cvx_opts = dict(solver=cvx.GUROBI,verbose=False,LogFile='')
        cvx_opts = dict(solver=cvx.ECOS,verbose=False)
        
        # Physical parameters
        self.omega = 2*np.pi/(24*3600+39*60+35) # [rad/s] Mars spin
        self.mass = 1700. # [kg] Lander mass
        self.g = 3.71 # [m/s^2] Mars surface gravity
        self.R = 3396.2e3 # [m] Mars radius
        #Tmax = 21500. # [N] Maximum thrust
        #amax = Tmax/self.mass; # [m/s^2] Maximum acceleration
        self.rho1 = [4,8]#[s_*amax for s_ in [0.2,0.6]] # [m/s^2] Smallest  acceleration
        self.rho2 = [8,12]#[s_*amax for s_ in [0.6,0.9]] # [m/s^2] Largest control acceleration
        self.gs = np.deg2rad(10) # [rad] Smallest glideslope angle
        
        # Boundary conditions
        r0 = np.array([1500.,agl])
        v0 = np.array([50.,-70.])
        rf = np.array([0.,0.])
        vf = np.array([0.,0.])
        
        # Thruster layout
        self.theta = [90.,20.] # [rad] Gimbal angles of [low,high] thrust modes
        cone_parameters = [dict(alpha=theta,roll=0,twod=True) for theta in self.theta]
        self.C = [tools.make_cone(**param) for param in cone_parameters]
        eps = np.sqrt(np.finfo(np.float64).eps) # Machine epsilon
        for i in range(len(self.C)):
            # Clean up small coefficients
            self.C[i][np.abs(self.C[i])<eps]=0
        self.M = len(self.C) # Number of thrusters
        self.K = 1 # How many thrusters can be simultaneously active
        
        # Setup dynamical system
        S = np.array([[0,1],[-1,0]])
        Ac = np.block([[np.zeros((2,2)),np.eye(2)],[self.omega**2*np.eye(2),2*self.omega*S]])
        Bc = np.row_stack([np.zeros((2,2)),np.eye(2)])
        wc = np.row_stack([np.zeros((3,1)),self.omega**2*self.R-self.g])
        self.A,self.B,self.w = Ac,Bc,wc
        nx,nu = Ac.shape[1], Bc.shape[1]
        A = cvx.Parameter(nx,nx)
        B = cvx.Parameter(nx,nu)
        w = cvx.Parameter(nx,1)
        
        # Scaling
        Dx = np.concatenate(np.abs([r0,v0]))
        Dx[Dx==0] = 1
        Dx[0] = r0[1]*np.tan(np.pi/2-self.gs)
        self.Dx = np.diag(Dx)
        self.Du = [np.diag([rho2 for _ in range(nu)]) for rho2 in self.rho2]
        self.tfmax = 100.
        
        # Optimization problem common parts
        self.N = 50 # Temporal solution
        x = [cvx.Parameter(nx)]+[self.Dx*cvx.Variable(nx) for _ in range(1,self.N+1)]
        xi = [cvx.Parameter()]+[cvx.Variable() for _ in range(1,self.N+1)]
        u = [[self.Du[i]*cvx.Variable(nu) for __ in range(self.N)] for i in range(self.M)]
        sigma = [cvx.Variable(self.N) for _ in range(self.M)]
        gamma = [cvx.Bool(self.N) if micp else cvx.Variable(self.N) for _ in range(self.M)]
        dt = cvx.Parameter()
        J2 = cvx.Parameter()
#        Q = np.diag([1.,0.,0.,0.])
        
        # Cost components
#        rdt = cvx.sqrt(dt)
        tf = dt*self.N
        ximax = self.tfmax*np.max(self.rho2)
        Dxi = la.inv(self.Dx)
        time_penalty = tf
        input_penalty = xi[-1]
        wx = 1e-3*self.tfmax
#        state_penalty = sum([cvx.quad_form(rdt*x[k],Dxi.dot(Q).dot(Dxi)) for k in range(self.N)])
        state_penalty = sum([cvx.abs(dt*Dxi[0,0]*x[k][0])+1e-4*cvx.abs(dt*Dxi[1,1]*x[k][1])
                             for k in range(self.N+1)])
        
        self.zeta = 0#1 if not micp else 0 # minimum time: 0
        cost_p2 = time_penalty+wx*state_penalty#time_penalty+wx*state_penalty
        cost_p3 = time_penalty # minimum time
        
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
                
        add_constraint([x[k+1] == A*x[k]+B*sum([u[i][k] for i in range(self.M)])+w for k in range(self.N)],'lambda')
        add_constraint([xi[k+1] == xi[k]+sum([dt*sigma[i][k] for i in range(self.M)]) for k in range(self.N)],'eta')
        x[0].value = np.concatenate([r0,v0])
        xi[0].value = 0
        add_constraint([x[-1] == np.concatenate([rf,vf])],'nu_xN')
        for i in range(self.M):
            add_constraint([cvx.norm2(u[i][k]) <= sigma[i][k] for k in range(self.N)],'beta_1_%d'%(i+1))
            add_constraint([gamma[i][k]*self.rho1[i] <= sigma[i][k] for k in range(self.N)],'beta_2_%d'%(i+1))
            add_constraint([sigma[i][k] <= gamma[i][k]*self.rho2[i] for k in range(self.N)],'beta_3_%d'%(i+1))
            if not micp:
                add_constraint([gamma[i][k] >= 0 for k in range(self.N)],'beta_4_%d'%(i+1))
                add_constraint([gamma[i][k] <= 1 for k in range(self.N)],'beta_5_%d'%(i+1))
            add_constraint([self.C[i]*u[i][k] <= 0 for k in range(self.N)],'beta_6_%d'%(i+1))
        add_constraint([sum([gamma[i][k] for i in range(self.M)]) <= self.K for k in range(self.N)],'beta_7')
        E_ = np.column_stack([np.eye(2),np.zeros((2,2))])
        gs = np.pi/2-self.gs
        add_constraint([np.array([0,1]).dot(E_)*x[k]>=cvx.norm2(E_*x[k])*np.cos(gs)
                        for k in range(self.N)],'lambda_gs')
        
        # Problem 2 oracle
        p2 = cvx.Problem(cvx.Minimize(cost_p2),self.constraints)
        
        def extract_variables():
            primal = dict()
            dual = dict()
            misc = dict()
            # Primal variables
            primal['x'] = np.column_stack([tools.cvx2arr(x[k]) for k in range(self.N+1)])
            primal['u'] = [np.column_stack([tools.cvx2arr(u[i][k]) for k in range(self.N)]) for i in range(self.M)]
            primal['sigma'] = [np.array([sigma[i][k].value for k in range(self.N)]) for i in range(self.M)]
            primal['gamma'] = [np.array([gamma[i][k].value for k in range(self.N)]) for i in range(self.M)]
            # Dual variables
            if not micp:
                for name in self.dual2idx.keys():
                    dual[name] = np.column_stack([tools.cvx2arr(self.constraints[k],dual=True) for k in self.dual2idx[name]])
                # Other (derived) variables
                misc['y'] = np.row_stack([self.Bd.T.dot(dual['lambda'][:,k]) for k in range(self.N)])
            return primal,dual,misc
        
        def problem2(tf):
            dt.value = tf/float(self.N)
            A.value,B.value = tools.discretize(Ac,Bc,dt.value)
            ___,w.value = tools.discretize(Ac,wc,dt.value)
            self.Ad = np.array(A.value)
            self.Bd = np.array(B.value)
            self.wd = np.array(w.value)
            
            t = np.array([k*dt.value for k in range(self.N+1)])
            
            try:
                J = p2.solve(**cvx_opts)
                solver_time = p2.solver_stats.solve_time
                if p2.status=='infeasible':
                    return p2.status,np.inf,t,None,None,None,solver_time
                else:
                    # All good, return solution
                    primal,dual,misc = extract_variables()
                    return p2.status,J,t,primal,dual,misc,solver_time
            except cvx.SolverError:
                return 'error',np.inf,t,None,None,None,0.
            
        self.problem2 = lambda tf: problem2(tf)
        
        # Problem 3 oracle
        p3 = cvx.Problem(cvx.Minimize(cost_p3),self.constraints+[cost_p2<=J2])
        
        def problem3(tf,J):
            dt.value = tf/float(self.N)
            J2.value = J
            
            A.value,B.value = tools.discretize(Ac,Bc,dt.value)
            ___,w.value = tools.discretize(Ac,wc,dt.value)
            self.Ad = np.array(A.value)
            self.Bd = np.array(B.value)
            self.wd = np.array(w.value)
            
            t = np.array([k*dt.value for k in range(self.N+1)])
            
            try:
                J = p3.solve(**cvx_opts)
                solver_time = p3.solver_stats.solve_time
                if p3.status=='infeasible':
                    return p3.status,np.inf,t,None,None,None,solver_time
                else:
                    # All good, return solution
                    primal,dual,misc = extract_variables()
                    cost_mintime = tf
                    return p3.status,cost_mintime,t,primal,dual,misc,solver_time
            except cvx.SolverError:
                return 'error',np.inf,t,None,None,None,0.
        
        self.problem3 = lambda tf,J2: problem3(tf,J2)
        
def save(pbm,J,t,primal,dual,misc,solver_time,filename):
    """
    Save the data.
    """
    pickle.dump(dict(rho1=pbm.rho1,rho2=pbm.rho2,N=pbm.N,M=pbm.M,K=pbm.K,
                     theta=pbm.theta,gs=pbm.gs,C=pbm.C,zeta=pbm.zeta,
                     J=J,t=t,primal=primal,dual=dual,misc=misc,solver_time=solver_time),open(filename,'wb'))

def solve_landing(h0):
    """
    Solve the rocket powered descent guidance problem with a 2-mode thruster.
    
    Parameters
    ----------
    h0 : float
        Initial altitude above ground level (AGL).
    """    
    #%% Lossless convexification solution
    
    rocket = Lander(agl=h0)
    J,t,primal,dual,misc,solver_time = lcvx.solve(rocket,[0.,100.],opt_tol=1e-4)
    save(rocket,J,t,primal,dual,misc,solver_time,'data/landing_lcvx_%dagl_debug.pkl'%(h0))
    landing_plots.plot_ifac20(data='data/landing_lcvx_%dagl_debug.pkl'%(h0),save_pdf=False,folder='%dagl/lcvx'%(h0))
    
    #%% Mixed-integer solution
    
# =============================================================================
#     if h0 > 650:
#         rocket = Lander(agl=h0,micp=True)
#         J,t,primal,dual,misc,solver_time = lcvx.solve(rocket,[0.,100.],opt_tol=1e-4)
#         save(rocket,J,t,primal,dual,misc,solver_time,'data/landing_micp_%dagl.pkl'%(h0))
#         landing_plots.plot_ifac20(data='data/landing_micp_%dagl.pkl'%(h0),save_pdf=True,folder='%dagl/micp'%(h0))
# =============================================================================

if __name__=='__main__':
    h0_list = [1000]#[650,750,850,1000,1200,1500,3000]
    for h0 in h0_list:
        print '======== Running: %d AGL'%(h0)
        solve_landing(h0)