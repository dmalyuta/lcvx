"""
Spacecraft docking to a spinning space station [1].

[1] https://www.youtube.com/watch?v=c4tPQYNpW9k

D. Malyuta -- ACL, University of Washington
B. Acikmese -- ACL, University of Washington

Copyright 2019 University of Washington. All rights reserved.
"""

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
        cvx_opts = dict(solver=cvx.ECOS,verbose=False,abstol=1e-8,max_iters=1000)
        
        # Boundary conditions
        r0 = np.array([5.,5.,1e2])
        v0 = np.array([0.,0.,0.])
        rf = np.array([0.,0.,0.])
        vf = np.array([0.,0.,0.])
        
        # Physical parameters
        omega = np.array([0.01*2*np.pi/60.,0.*2*np.pi/60.,1.*2*np.pi/60.]) # [rad/s] Space station spin
        self.rho1 = 0.5e-2 # [m/s^2] Smallest control acceleration
        self.rho2 = 0.5e-1 # [m/s^2] Largest control acceleration
        
        # RCS layout
        cone_angle = 0.001 # [deg] Thruster cone angle
        theta = phi = 30 # [deg] Basic pitch, roll
#        cone_parameters = [dict(alpha=cone_angle,roll=phi,pitch=theta,yaw=0),
#                           dict(alpha=cone_angle,roll=-phi,pitch=theta,yaw=0),
#                           dict(alpha=cone_angle,roll=-phi,pitch=-theta,yaw=0),
#                           dict(alpha=cone_angle,roll=phi,pitch=-theta,yaw=0),
#                           dict(alpha=cone_angle,roll=phi+(180-2*phi),pitch=-theta,yaw=0),
#                           dict(alpha=cone_angle,roll=-phi-(180-2*phi),pitch=-theta,yaw=0),
#                           dict(alpha=cone_angle,roll=-phi-(180-2*phi),pitch=theta,yaw=0),
#                           dict(alpha=cone_angle,roll=phi+(180-2*phi),pitch=theta,yaw=0),
#                           dict(alpha=cone_angle,roll=0,pitch=0,yaw=0),
#                           dict(alpha=cone_angle,roll=180,pitch=0,yaw=0)]
        cone_parameters = [#dict(alpha=cone_angle,roll=0,pitch=0,yaw=0),
                           dict(alpha=cone_angle,roll=180,pitch=0,yaw=0),
                           dict(alpha=cone_angle,roll=90,pitch=0,yaw=0),
                           dict(alpha=cone_angle,roll=-90,pitch=0,yaw=0),
                           dict(alpha=cone_angle,roll=0,pitch=90,yaw=0),
                           dict(alpha=cone_angle,roll=0,pitch=-90,yaw=0)]
        self.C = [tools.make_cone(**param) for param in cone_parameters]
        self.Cn = [tools.make_cone(normal=True,**param) for param in cone_parameters]
        self.M = len(self.C) # Number of thrusters
        K = 1 # How many thrusters can be simultaneously active
        
        # Setup dynamical system
        S = lambda w: np.array([[0,-w[2],w[1]],[w[2],0,-w[0]],[-w[1],w[0],0]])
        Ac = np.block([[np.zeros((3,3)),np.eye(3)],[-mpow(S(omega),2),-2*S(omega)]])
        Bc = np.row_stack([np.zeros((3,3)),np.eye(3)])
        
        # Scaling
        nx,nu = Ac.shape[1], Bc.shape[1]
        Dx = np.concatenate(np.abs([r0,v0]))
        Dx[Dx==0] = 1
        Dx = np.diag(Dx)
        Du = np.diag([self.rho2 for _ in range(nu)])
        
        # Optimization problem common parts
        self.N = 100 # Temporal solution
        x = [Dx*cvx.Variable(nx) for _ in range(self.N+1)]
        xi = cvx.Variable(self.N+1)
        u = [[Du*cvx.Variable(nu) for __ in range(self.N)] for _ in range(self.M)]
        sigma = [cvx.Variable(self.N) for _ in range(self.M)]
        gamma = [cvx.Variable(self.N) for _ in range(self.M)]
        
        constraints = []
        constraints += [x[0] == np.concatenate([r0,v0]),
                        x[-1] == np.concatenate([rf,vf]),
                        xi[0] == 0]
        for i in range(self.M):
            constraints += [cvx.norm2(u[i][k]) <= sigma[i][k] for k in range(self.N)]
            constraints += [gamma[i][k]*self.rho1 <= sigma[i][k] for k in range(self.N)]
            constraints += [sigma[i][k] <= gamma[i][k]*self.rho2 for k in range(self.N)]
            constraints += [gamma[i][k] >= 0 for k in range(self.N)]
            constraints += [gamma[i][k] <= 1 for k in range(self.N)]
            constraints += [self.C[i]*u[i][k] <= 0 for k in range(self.N)]
        constraints += [sum([gamma[i][k] for i in range(self.M)]) <= K for k in range(self.N)]
                
        # Problem 2 oracle
        def problem2(tf):
            dt = tf/float(self.N)
            A,B = tools.discretize(Ac,Bc,dt)
            
            self.zeta = 0
            cost = cvx.Minimize(tf)
            extra_constraints = []
            extra_constraints += [x[k+1] == A*x[k]+B*sum([u[i][k] for i in range(self.M)]) for k in range(self.N)]
            extra_constraints += [xi[k+1] == xi[k]+dt*sum([sigma[i][k] for i in range(self.M)]) for k in range(self.N)]
            
            p2 = cvx.Problem(cost,extra_constraints+constraints)
            
            t = np.array([k*dt for k in range(self.N+1)])
            try:
                J = p2.solve(**cvx_opts)
                if p2.status=='infeasible':
                    return p2.status,J,t,None,None,None
                else:
                    # All good, return solution
                    primal = dict()
                    dual = dict()
                    misc = dict()
                    primal['x'] = np.column_stack([tools.cvx2arr(x[k]) for k in range(self.N+1)])
                    primal['u'] = [np.column_stack([tools.cvx2arr(u[i][k]) for k in range(self.N)]) for i in range(self.M)]
                    primal['sigma'] = [np.array([sigma[i][k].value for k in range(self.N)]) for i in range(self.M)]
                    dual['lambda'] = np.column_stack([np.array(extra_constraints[k].dual_value.T).flatten() for k in range(self.N)])
                    misc['y'] = np.row_stack([B.T.dot(dual['lambda'][:,k]) for k in range(self.N)])
                    return p2.status,J,t,primal,dual,misc
            except cvx.SolverError:
                return 'cvx.SolverError',np.inf,t,None,None,None
            
        self.problem2 = lambda tf: problem2(tf)
        
cooper = Docker()
#import sys
#sys.exit()
#J,t,primal,dual,misc = lcvx.solve(cooper,[0.,1000.],opt_tol=1e-3)
J,t,primal,dual,misc = lcvx.solve(cooper,[480]*2,opt_tol=1e-2) #293.48

#%%

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
ax.axhline(cooper.rho1*3600.**2/1e3,color='gray',linestyle='--',linewidth=norm_linewidth,zorder=0,label='$\\rho_1$, $\\rho_2$')
ax.axhline(cooper.rho2*3600.**2/1e3,color='gray',linestyle='--',linewidth=norm_linewidth,zorder=0)
ax.autoscale(tight=True)
ax.set_ylim([0,cooper.rho2*1.1*3600.**2/1e3])
ax.set_ylabel('$\|u(t)\|_2$')
#ax.legend(prop={'size':8})
ax.get_xaxis().set_ticklabels([])
#ax.tick_params(axis ='x', direction='in') 
for i in range(cooper.M):
    ax = fig.add_subplot(1+cooper.M,1,i+2)
    ax.plot(t[:-1],[la.norm(primal['u'][i][:,k]) for k in range(cooper.N)],color='red',linewidth=norm_linewidth,label='$||u_%d||_2$'%(i+1))
    ax.plot(t[:-1],np.array(primal['sigma'][i]),color='orange',linestyle='none',marker='.',markersize=sigma_marker_size,label='$\sigma_%d$'%(i+1))
    ax.axhline(cooper.rho1,color='gray',linestyle='--',linewidth=norm_linewidth,zorder=0,label='$\\rho_1$, $\\rho_2$')
    ax.axhline(cooper.rho2,color='gray',linestyle='--',linewidth=norm_linewidth,zorder=0)
    ax.autoscale(tight=True)
    ax.set_ylim([0,cooper.rho2*1.1])
    ax.set_ylabel('$\|u_%d(t)\|_2$'%(i+1))
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
#%%
fig = plt.figure(5,figsize=(8,10))
plt.clf()
ax = fig.add_subplot(111)
for i in range(cooper.M):
    ax.plot(t[:-1],norm_z[i],marker='.',label='%d'%(i+1),linewidth=2 if i==0 else 1)
ax.legend()
ax.set_xlabel('Time [s]')
ax.set_ylabel('Projection onto $U_i$')