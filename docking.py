"""
Spacecraft docking to a spinning space station [1].

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
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import lcvx
import tools

class Docker(lcvx.Problem,object):
    def __init__(self):
        super(Docker, self).__init__()
        cvx_opts = dict(solver=cvx.ECOS,verbose=False,abstol=1e-8,max_iters=100)
        
        # Boundary conditions
        r0 = np.array([5.,5.,1e2])
        v0 = np.array([0.,0.,0.])
        rf = np.array([0.,0.,0.])
        vf = np.array([0.,0.,-1e-2])
        
        # Physical parameters
        omega = np.array([0.*2*np.pi/60.,0.*2*np.pi/60.,1.*2*np.pi/60.]) # [rad/s] Space station spin
        self.rho1 = 1e-3#0.5e-2 # [m/s^2] Smallest control acceleration
        self.rho2 = 1e-2#0.5e-1 # [m/s^2] Largest control acceleration
        
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
        Ac = np.block([[np.zeros((3,3)),np.eye(3)],[-mpow(S(omega),2),-2*S(omega)]])
        Bc = np.row_stack([np.zeros((3,3)),np.eye(3)])
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

pickle.dump(dict(rho1=cooper.rho1,rho2=cooper.rho2,M=cooper.M,
                 K=cooper.K,C=cooper.C,t=t,primal=primal,dual=dual,misc=misc),
            open('solution.pkl','wb'))

#%% Plot the solution

# Thruster layout
fig = plt.figure(1)
plt.clf()
ax = fig.add_subplot(111, projection='3d')
for i in range(len(cooper.C)):
    nhat = -sum(cooper.C[i])/la.norm(sum(cooper.C[i]))
    ax.plot([0,nhat[0]],[0,nhat[1]],[0,nhat[2]])
    ax.text(nhat[0],nhat[1],nhat[2],'%d'%(i+1))
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.axis('equal')

# State trajectory
norm_linewidth=1
fig = plt.figure(2,figsize=(9,4))
plt.clf()
# Position projections
ax = fig.add_subplot(271)
ax.plot(t,primal['x'][0],color='red',linewidth=norm_linewidth)
ax.plot(t,primal['x'][1],color='green',linewidth=norm_linewidth)
ax.plot(t,primal['x'][2],color='blue',linewidth=norm_linewidth)
ax.set_xlabel('Time [s]')
ax.set_ylabel('Position [m]')
ax.autoscale(tight=True)
ax = fig.add_subplot(272)
ax.plot(primal['x'][0],primal['x'][1],color='black',linewidth=norm_linewidth)
ax.set_xlabel('$x$ [m]')
ax.set_ylabel('$y$ [m]')
ax.autoscale(tight=True)
ax = fig.add_subplot(273)
ax.plot(primal['x'][0],primal['x'][2],color='black',linewidth=norm_linewidth)
ax.set_xlabel('$x$ [m]')
ax.set_ylabel('$z$ [m]')
ax.autoscale(tight=True)
ax = fig.add_subplot(274)
ax.plot(primal['x'][1],primal['x'][2],color='black',linewidth=norm_linewidth)
ax.set_xlabel('$y$ [m]')
ax.set_ylabel('$z$ [m]')
ax.autoscale(tight=True)
# Velocity projections
ax = fig.add_subplot(278)
ax.plot(t,primal['x'][3],color='red',linewidth=norm_linewidth)
ax.plot(t,primal['x'][4],color='green',linewidth=norm_linewidth)
ax.plot(t,primal['x'][5],color='blue',linewidth=norm_linewidth)
ax.set_xlabel('Time [s]')
ax.set_ylabel('Velocity [m/s]')
ax.autoscale(tight=True)
ax = fig.add_subplot(279)
ax.plot(primal['x'][3],primal['x'][4],color='black',linewidth=norm_linewidth)
ax.set_xlabel('$v_x$ [m/s]')
ax.set_ylabel('$v_y$ [m/s]')
ax.autoscale(tight=True)
ax = fig.add_subplot(2,7,10)
ax.plot(primal['x'][3],primal['x'][5],color='black',linewidth=norm_linewidth)
ax.set_xlabel('$v_x$ [m/s]')
ax.set_ylabel('$v_z$ [m/s]')
ax.autoscale(tight=True)
ax = fig.add_subplot(2,7,11)
ax.plot(primal['x'][4],primal['x'][5],color='black',linewidth=norm_linewidth)
ax.set_xlabel('$v_y$ [m/s]')
ax.set_ylabel('$v_z$ [m/s]')
ax.autoscale(tight=True)
plt.tight_layout()
# 3D trajectory
v_scale = 0.7e1
u_scale = 0.35e3
ax = fig.add_subplot(122, projection='3d')
#ax.axis('equal')
ax.plot(primal['x'][0],primal['x'][1],primal['x'][2],color='black',label='position')
for k in range(cooper.N+1):
    ax.plot([primal['x'][0,k],(primal['x'][0,k]+v_scale*primal['x'][3,k])],
            [primal['x'][1,k],(primal['x'][1,k]+v_scale*primal['x'][4,k])],
            [primal['x'][2,k],(primal['x'][2,k]+v_scale*primal['x'][5,k])],
            color='red',label='velocity' if k==0 else None,linewidth=0.2)
sum_u = sum(primal['u'])
for k in range(cooper.N):
    ax.plot([primal['x'][0,k],(primal['x'][0,k]-u_scale*sum_u[0,k])],
            [primal['x'][1,k],(primal['x'][1,k]-u_scale*sum_u[1,k])],
            [primal['x'][2,k],(primal['x'][2,k]-u_scale*sum_u[2,k])],
            color='blue',label='thrust' if k==0 else None,linewidth=0.2)
ax.set_xlabel('$x$ [m]')
ax.set_ylabel('$y$ [m]')
ax.set_zlabel('$z$ [m]')
ax.xaxis.pane.set_edgecolor('black')
ax.yaxis.pane.set_edgecolor('black')
ax.zaxis.pane.set_edgecolor('black')
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
#ax.legend()

# Input trajectory
sigma_marker_size = 4
norm_linewidth=1
fig = plt.figure(3,figsize=(5,6))
plt.clf()
ax = fig.add_subplot(1+cooper.M,1,1)
ax.plot(t[:-1],[sum([la.norm(primal['u'][i][:,k]) for i in range(cooper.M)]) for k in range(cooper.N)],color='gray',linestyle='--',linewidth=norm_linewidth,label='$\sum_{i=1}^M||u_i||_2$')
ax.plot(t[:-1],np.array(sum(primal['sigma'])),color='orange',linestyle='none',marker='.',markersize=sigma_marker_size,label='$\sum_{i=1}^M\sigma_i$')
ax.axhline(cooper.rho1,color='gray',linestyle='--',linewidth=norm_linewidth,zorder=0,label='$\\rho_1$, $\\rho_2$')
for i in range(cooper.K):
    ax.axhline((i+1)*cooper.rho2,color='gray',linestyle='--',linewidth=norm_linewidth,zorder=0)
ax.autoscale(tight=True)
ax.set_ylim([0,cooper.rho2*cooper.K*1.1])
ax.set_ylabel('$\|u(t)\|_2$')
#ax.legend(prop={'size':8})
ax.get_xaxis().set_ticklabels([])
#ax.tick_params(axis ='x', direction='in') 
for i in range(cooper.M):
    ax = fig.add_subplot(1+cooper.M,1,i+2)
    ax.plot(t[:-1],[la.norm(primal['u'][i][:,k]) for k in range(cooper.N)],color='red',linewidth=norm_linewidth,label='$||u_%d||_2$'%(i+1))
    ax.plot(t[:-1],np.array(primal['sigma'][i]),color='orange',linestyle='none',marker='.',markersize=sigma_marker_size,label='$\sigma_%d$'%(i+1))
    ax.plot(t[:-1],np.array(primal['gamma'][i])*cooper.rho1,color='green',linestyle=':',linewidth=1,label='$\gamma_%d\\rho_1$'%(i+1))
    ax.plot(t[:-1],np.array(primal['gamma'][i])*cooper.rho2,color='blue',linestyle=':',linewidth=1,label='$\gamma_%d\\rho_2$'%(i+1))
    ax.axhline(cooper.rho1,color='gray',linestyle='--',linewidth=norm_linewidth,zorder=0,label='$\\rho_1$, $\\rho_2$')
    ax.axhline(cooper.rho2,color='gray',linestyle='--',linewidth=norm_linewidth,zorder=0)
    ax.autoscale(tight=True)
    ax.set_ylim([0,cooper.rho2*1.1])
    ax.set_ylabel('$\|u_{%d}(t)\|_2$'%(i+1))
    if i==cooper.M-1:
        ax.set_xlabel('Time [s]')
    else:
        pass#ax.get_xaxis().set_ticklabels([])
    #ax.legend(prop={'size':8})
    #ax.tick_params(axis ='x', direction='in') 
plt.tight_layout()
plt.subplots_adjust(hspace=0.2)

# Dual variable y trajectory
fig = plt.figure(4)
plt.clf()
ax = fig.add_subplot(321)
ax.plot(misc['y'][:,0],misc['y'][:,1],color='black')
ax.set_xlabel('x-component')
ax.set_ylabel('y-component')
ax = fig.add_subplot(323)
ax.plot(misc['y'][:,0],misc['y'][:,2],color='black')
ax.set_xlabel('x-component')
ax.set_ylabel('z-component')
ax = fig.add_subplot(325)
ax.plot(misc['y'][:,1],misc['y'][:,2],color='black')
ax.set_xlabel('y-component')
ax.set_ylabel('z-component')
ax = fig.add_subplot(122,projection='3d')
ax.plot(misc['y'][:,0],misc['y'][:,1],misc['y'][:,2],color='black')
ax.set_xlabel('x-component')
ax.set_ylabel('y-component')
ax.set_zlabel('z-component')
plt.tight_layout()

# Dual variable projections onto pointing sets
norm_z = [np.array([la.norm(tools.project(misc['y'][k],cooper.C[i])) for k in range(cooper.N)]) for i in range(cooper.M)]
fig = plt.figure(5,figsize=(8,10))
plt.clf()
ax = fig.add_subplot(111)
for i in range(cooper.M):
    ax.plot(t[:-1],norm_z[i],marker='.',label='%d'%(i+1),linewidth=2 if i==0 else 1)
ax.legend()
ax.set_xlabel('Time [s]')
ax.set_ylabel('Projection onto $U_i$')

#%% Plot dual problem constraints

ik2idx = lambda i,k: k+i*cooper.N

fig = plt.figure(6)
plt.clf()
ax = fig.add_subplot(111)
constraint_values = [dual['lambda_rho2'][:,ik2idx(i,k)]-dual['lambda_rho1'][:,ik2idx(i,k)]-dual['lambda_sigma'][:,ik2idx(i,k)] for i in range(cooper.M) for k in range(cooper.N)]
ax.plot(constraint_values)
ax.set_title('$\lambda_k^{\\rho_2 i}-\lambda_k^{\\rho_1 i}-\lambda_k^{\\sigma i} = 0$')
ax.set_xlabel('$i,k$ indices')
ax.set_ylabel('Value')

fig = plt.figure(7)
plt.clf()
ax = fig.add_subplot(111)
constraint_values = [dual['lambda_rho1'][:,ik2idx(i,k)]*cooper.rho1-dual['lambda_rho2'][:,ik2idx(i,k)]*cooper.rho2+dual['lambda_gamma_high'][:,ik2idx(i,k)]-dual['lambda_gamma_low'][:,ik2idx(i,k)]+dual['lambda_sum_gamma'][:,k] for i in range(cooper.M) for k in range(cooper.N)]
ax.plot(constraint_values)
ax.set_title('$\lambda_k^{\\rho_1 i}\\rho_1-\lambda_k^{\\rho_2 i}\\rho_2+\lambda_k^{\\bar\gamma i}-\lambda_k^{\underbar{\gamma} i}+\lambda_k^{\\sum\gamma} = 0$')
ax.set_xlabel('$i,k$ indices')
ax.set_ylabel('Value')

fig = plt.figure(8)
plt.clf()
ax = fig.add_subplot(111)
constraint_values = [dual['nu_x'][:,k-1]-cooper.Ad.T.dot(dual['nu_x'][:,k]) for k in range(1,cooper.N)]
ax.plot(constraint_values)
ax.set_title('$\\nu_{k-1}^x-A^T\\nu_{k}^x=0$')
ax.set_xlabel('$i,k$ indices')
ax.set_ylabel('Value')

fig = plt.figure(9)
plt.clf()
ax = fig.add_subplot(111)
constraint_values = dual['nu_x'][:,cooper.N-1]+dual['nu_xN'].flatten()
ax.plot(constraint_values)
ax.set_title('$\\nu_{N-1}^x+\\nu^{x_N}=0$')
ax.set_xlabel('$i,k$ indices')
ax.set_ylabel('Value')

fig = plt.figure(10)
plt.clf()
ax = fig.add_subplot(211)
constraint_values = [la.norm(-cooper.C[i].T.dot(dual['lambda_u'][:,ik2idx(i,k)])+cooper.Bd.T.dot(dual['nu_x'][:,k]))-dual['lambda_sigma'][:,ik2idx(i,k)] for i in range(cooper.M) for k in range(cooper.N)]
u_norm_values = np.concatenate([la.norm(primal['u'][i],axis=0) for i in range(len(primal['u']))])
ax.plot(constraint_values)
ax.fill_between(range(u_norm_values.size),(u_norm_values>1e-6)*np.min(constraint_values),
                alpha=0.2,color='red',linewidth=0,step='mid',label='$\|\|u_k^i\|\|_2>0$')
ax.set_title('$\|\| -C_i^T\lambda_k^{u i}+B^T\\nu_k^x \|\|_2- \lambda_k^{\\sigma i}\leq 0$')
ax.set_ylabel('Value')
ax.legend()
ax = fig.add_subplot(212)
constraint_values = [la.norm(-cooper.C[i].T.dot(dual['lambda_u'][:,ik2idx(i,k)])+cooper.Bd.T.dot(dual['nu_x'][:,k])) for i in range(cooper.M) for k in range(cooper.N)]
u_norm_values = np.concatenate([la.norm(primal['u'][i],axis=0) for i in range(len(primal['u']))])
ax.plot(constraint_values,label='lhs')
ax.plot(dual['lambda_sigma'].T,label='rhs')
ax.fill_between(range(u_norm_values.size),(u_norm_values>1e-6)*np.max(dual['lambda_sigma']),
                alpha=0.2,color='red',linewidth=0,step='mid',label='$\|\|u_k^i\|\|_2>0$')
ax.set_xlabel('$i,k$ indices')
ax.set_ylabel('Value')
ax.legend()

fig = plt.figure(11)
plt.clf()
ax = fig.add_subplot(211)
constraint_values = [cooper.rho2*la.norm(-cooper.C[i].T.dot(dual['lambda_u'][:,ik2idx(i,k)])+cooper.Bd.T.dot(dual['nu_x'][:,k]))-dual['lambda_gamma_high'][:,ik2idx(i,k)]+dual['lambda_gamma_low'][:,ik2idx(i,k)] for i in range(cooper.M) for k in range(cooper.N)]
difference = np.empty((cooper.M,cooper.N))
for i in range(cooper.M):
    difference[i] = dual['lambda_sum_gamma'].flatten()-np.array(constraint_values).flatten()[i*cooper.N:(i+1)*cooper.N]
idxs = np.argsort(difference,axis=0)
for i in range(cooper.M):
    diffs = difference[i].copy()
    diffs[np.all(idxs[:cooper.K]!=i,axis=0)] = np.nan
    ax.plot(t[:-1],diffs,marker='.',label='argmin: $i=%d$'%(i+1))
ax.set_title('$\\rho_2\|\| -C_i^T\lambda_k^{u i}+B^T\\nu_k^x \|\|_2-\lambda_k^{\\bar\gamma i}+\lambda_k^{\\underbar\gamma i}\leq \lambda_k^{\sum\gamma}$ $\\forall i=1,\dots,M,$ $k=0,\dots,N-1$')
ax.set_ylabel('RHS-LHS minimum value')
ax.legend()
ax = fig.add_subplot(212)
for i in range(cooper.M):
    difference[i] = dual['lambda_sum_gamma'].flatten()-np.array(constraint_values).flatten()[i*cooper.N:(i+1)*cooper.N]
    ax.plot(t[:-1],difference[i],label='$i=%d$'%(i+1))
ax.set_xlabel('Time [s]')
ax.set_ylabel('LHS value')
ax.legend()

#%% Other plots

fig = plt.figure(12)
plt.clf()
ax = fig.add_subplot(111)
t_range = np.linspace(150,350,20)
J = tools.cost_profile(oracle = lambda tf: cooper.problem2(tf)[1],t_range = t_range)
ax.plot(t_range,J)
ax.set_xlabel('Final time [s]')
ax.set_ylabel('Optimal cost value')