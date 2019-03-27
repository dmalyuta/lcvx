"""
Different work-in-progress plots.

D. Malyuta -- ACL, University of Washington
B. Acikmese -- ACL, University of Washington

Copyright 2019 University of Washington. All rights reserved.
"""

import pickle
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import tools

#%% Load data

solution = pickle.load(open('solution.pkl','rb'))
t,primal,dual,misc = solution['t'],solution['primal'],solution['dual'],solution['misc']
rho1,rho2,N,M,K,C = solution['rho1'],solution['rho2'],solution['N'],solution['M'],solution['K'],solution['C']
A,B,J,Jt = solution['A'],solution['B'],solution['J'],solution['Jt']

#%% Plot the solution

# Thruster layout
fig = plt.figure(1)
plt.clf()
ax = fig.add_subplot(111, projection='3d')
for i in range(len(C)):
    nhat = -sum(C[i])/la.norm(sum(C[i]))
    ax.plot([0,nhat[0]],[0,nhat[1]],[0,nhat[2]])
    ax.text(nhat[0],nhat[1],nhat[2],'%d'%(i+1))
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.axis('equal')

# State trajectory
norm_linewidth=1
fig = plt.figure(2,figsize=(16,5))
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
for k in range(N+1):
    ax.plot([primal['x'][0,k],(primal['x'][0,k]+v_scale*primal['x'][3,k])],
            [primal['x'][1,k],(primal['x'][1,k]+v_scale*primal['x'][4,k])],
            [primal['x'][2,k],(primal['x'][2,k]+v_scale*primal['x'][5,k])],
            color='red',label='velocity' if k==0 else None,linewidth=0.2)
sum_u = sum(primal['u'])
for k in range(N):
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

#%%

# Input trajectory
sigma_marker_size = 4
norm_linewidth=1
fig = plt.figure(3,figsize=(5,6))
plt.clf()
ax = fig.add_subplot(1+M,1,1)
ax.plot(t[:-1],[sum([la.norm(primal['u'][i][:,k]) for i in range(M)]) for k in range(N)],color='gray',linestyle='--',linewidth=norm_linewidth,label='$\sum_{i=1}^M||u_i||_2$')
ax.plot(t[:-1],np.array(sum(primal['sigma'])),color='orange',linestyle='none',marker='.',markersize=sigma_marker_size,label='$\sum_{i=1}^M\sigma_i$')
ax.axhline(rho1,color='gray',linestyle='--',linewidth=norm_linewidth,zorder=0,label='$\\rho_1$, $\\rho_2$')
for i in range(K):
    ax.axhline((i+1)*rho2,color='gray',linestyle='--',linewidth=norm_linewidth,zorder=0)
ax.autoscale(tight=True)
ax.set_ylim([0,rho2*K*1.1])
ax.set_ylabel('$\|u(t)\|_2$')
#ax.legend(prop={'size':8})
ax.get_xaxis().set_ticklabels([])
#ax.tick_params(axis ='x', direction='in') 
for i in range(M):
    ax = fig.add_subplot(1+M,1,i+2)
    ax.plot(t[:-1],[la.norm(primal['u'][i][:,k]) for k in range(N)],color='red',linewidth=norm_linewidth,label='$||u_%d||_2$'%(i+1))
    ax.plot(t[:-1],np.array(primal['sigma'][i]),color='orange',linestyle='none',marker='.',markersize=sigma_marker_size,label='$\sigma_%d$'%(i+1))
    ax.plot(t[:-1],np.array(primal['gamma'][i])*rho1,color='green',linestyle=':',linewidth=1,label='$\gamma_%d\\rho_1$'%(i+1))
    ax.plot(t[:-1],np.array(primal['gamma'][i])*rho2,color='blue',linestyle=':',linewidth=1,label='$\gamma_%d\\rho_2$'%(i+1))
    ax.axhline(rho1,color='gray',linestyle='--',linewidth=norm_linewidth,zorder=0,label='$\\rho_1$, $\\rho_2$')
    ax.axhline(rho2,color='gray',linestyle='--',linewidth=norm_linewidth,zorder=0)
    ax.autoscale(tight=True)
    ax.set_ylim([0,rho2*1.1])
    ax.set_ylabel('$\|u_{%d}(t)\|_2$'%(i+1))
    if i==M-1:
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
norm_z = [np.array([la.norm(tools.project(misc['y'][k],C[i])) for k in range(N)]) for i in range(M)]
fig = plt.figure(5,figsize=(8,10))
plt.clf()
ax = fig.add_subplot(111)
for i in range(M):
    ax.plot(t[:-1],norm_z[i],marker='.',label='%d'%(i+1),linewidth=2 if i==0 else 1)
ax.legend()
ax.set_xlabel('Time [s]')
ax.set_ylabel('Projection onto $U_i$')

#%% Plot dual problem constraints

ik2idx = lambda i,k: k+i*N

fig = plt.figure(6)
plt.clf()
ax = fig.add_subplot(111)
constraint_values = [dual['lambda_rho2'][:,ik2idx(i,k)]-dual['lambda_rho1'][:,ik2idx(i,k)]-dual['lambda_sigma'][:,ik2idx(i,k)] for i in range(M) for k in range(N)]
ax.plot(constraint_values)
ax.set_title('$\lambda_k^{\\rho_2 i}-\lambda_k^{\\rho_1 i}-\lambda_k^{\\sigma i} = 0$')
ax.set_xlabel('$i,k$ indices')
ax.set_ylabel('Value')

fig = plt.figure(7)
plt.clf()
ax = fig.add_subplot(111)
constraint_values = [dual['lambda_rho1'][:,ik2idx(i,k)]*rho1-dual['lambda_rho2'][:,ik2idx(i,k)]*rho2+dual['lambda_gamma_high'][:,ik2idx(i,k)]-dual['lambda_gamma_low'][:,ik2idx(i,k)]+dual['lambda_sum_gamma'][:,k] for i in range(M) for k in range(N)]
ax.plot(constraint_values)
ax.set_title('$\lambda_k^{\\rho_1 i}\\rho_1-\lambda_k^{\\rho_2 i}\\rho_2+\lambda_k^{\\bar\gamma i}-\lambda_k^{\underbar{\gamma} i}+\lambda_k^{\\sum\gamma} = 0$')
ax.set_xlabel('$i,k$ indices')
ax.set_ylabel('Value')

fig = plt.figure(8)
plt.clf()
ax = fig.add_subplot(111)
constraint_values = [dual['nu_x'][:,k-1]-A.T.dot(dual['nu_x'][:,k]) for k in range(1,N)]
ax.plot(constraint_values)
ax.set_title('$\\nu_{k-1}^x-A^T\\nu_{k}^x=0$')
ax.set_xlabel('$i,k$ indices')
ax.set_ylabel('Value')

fig = plt.figure(9)
plt.clf()
ax = fig.add_subplot(111)
constraint_values = dual['nu_x'][:,N-1]+dual['nu_xN'].flatten()
ax.plot(constraint_values)
ax.set_title('$\\nu_{N-1}^x+\\nu^{x_N}=0$')
ax.set_xlabel('$i,k$ indices')
ax.set_ylabel('Value')

fig = plt.figure(10)
plt.clf()
ax = fig.add_subplot(211)
constraint_values = [la.norm(-C[i].T.dot(dual['lambda_u'][:,ik2idx(i,k)])+B.T.dot(dual['nu_x'][:,k]))-dual['lambda_sigma'][:,ik2idx(i,k)] for i in range(M) for k in range(N)]
u_norm_values = np.concatenate([la.norm(primal['u'][i],axis=0) for i in range(len(primal['u']))])
ax.plot(constraint_values)
ax.fill_between(range(u_norm_values.size),(u_norm_values>1e-6)*np.min(constraint_values),
                alpha=0.2,color='red',linewidth=0,step='mid',label='$\|\|u_k^i\|\|_2>0$')
ax.set_title('$\|\| -C_i^T\lambda_k^{u i}+B^T\\nu_k^x \|\|_2- \lambda_k^{\\sigma i}\leq 0$')
ax.set_ylabel('Value')
ax.legend()
ax = fig.add_subplot(212)
constraint_values = [la.norm(-C[i].T.dot(dual['lambda_u'][:,ik2idx(i,k)])+B.T.dot(dual['nu_x'][:,k])) for i in range(M) for k in range(N)]
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
constraint_values = [rho2*la.norm(-C[i].T.dot(dual['lambda_u'][:,ik2idx(i,k)])+B.T.dot(dual['nu_x'][:,k]))-dual['lambda_gamma_high'][:,ik2idx(i,k)]+dual['lambda_gamma_low'][:,ik2idx(i,k)] for i in range(M) for k in range(N)]
difference = np.empty((M,N))
for i in range(M):
    difference[i] = dual['lambda_sum_gamma'].flatten()-np.array(constraint_values).flatten()[i*N:(i+1)*N]
idxs = np.argsort(difference,axis=0)
for i in range(M):
    diffs = difference[i].copy()
    diffs[np.all(idxs[:K]!=i,axis=0)] = np.nan
    ax.plot(t[:-1],diffs,marker='.',label='argmin: $i=%d$'%(i+1))
ax.set_title('$\\rho_2\|\| -C_i^T\lambda_k^{u i}+B^T\\nu_k^x \|\|_2-\lambda_k^{\\bar\gamma i}+\lambda_k^{\\underbar\gamma i}\leq \lambda_k^{\sum\gamma}$ $\\forall i=1,\dots,M,$ $k=0,\dots,N-1$')
ax.set_ylabel('RHS-LHS minimum value')
ax.legend()
ax = fig.add_subplot(212)
for i in range(M):
    difference[i] = dual['lambda_sum_gamma'].flatten()-np.array(constraint_values).flatten()[i*N:(i+1)*N]
    ax.plot(t[:-1],difference[i],label='$i=%d$'%(i+1))
ax.set_xlabel('Time [s]')
ax.set_ylabel('LHS value')
ax.legend()

#%% Other plots

fig = plt.figure(12)
plt.clf()
ax = fig.add_subplot(111)
ax.plot(Jt,J)
ax.set_xlabel('Final time [s]')
ax.set_ylabel('Optimal cost value')
