"""
Plots for the paper for docking.

D. Malyuta -- ACL, University of Washington
B. Acikmese -- ACL, University of Washington

Copyright 2019 University of Washington. All rights reserved.
"""

import os
import pickle
import numpy as np
import seaborn as sns
import numpy.linalg as la
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D

def plot_automatica19(data,in_inertial=True,save_pdf=False,folder=''):
    matplotlib.rc('font',**{'family':'serif','serif':['DejaVu Sans']})
    matplotlib.rc('text', usetex=True)
    
    frame = 'inertial' if in_inertial else 'rotating'
    
    #%% Load data
    
    solution = pickle.load(open(data,'rb'))
    t,primal,misc = solution['t'],solution['primal'],solution['misc']
    rho1,rho2,M,K,C = solution['rho1'],solution['rho2'],solution['M'],solution['K'],solution['C']
    
    print 'N: %d'%(solution['N'])
    print 'Solver time: %.1e s'%(solution['solver_time'])
    print 'Time of flight: %d s'%(t[-1])
    
    if save_pdf and not os.path.isdir('figures/%s'%(folder)):
        os.makedirs('figures/%s'%(folder))

    #%% Trajectory
    
    matplotlib.rcParams.update({'font.size': 18})
    
    xtraj = primal['x_inertial'] if in_inertial else primal['x']
    utraj = primal['u_inertial'] if in_inertial else primal['u']
    
    init3d_style = dict(marker='.',markersize=7,color='green',zorder=100)
    init2d_style = dict(marker='.',markersize=7,color='green',zorder=90)
    traj3d_style = dict(linewidth=2,color='black',zorder=99)
    vel3d_style = dict(linewidth=0.25,color='red',zorder=99)
    accel3d_style = dict(linewidth=0.25,color='blue',zorder=99)
    traj2d_style = dict(linewidth=1,color='gray')
    vel2d_style = dict(linewidth=0.25,color='red',alpha=0.5)
    accel2d_style = dict(linewidth=0.25,color='blue',alpha=0.5)
    vel_scale = (25 if in_inertial else 6)*np.eye(3)
    accel_scale = 3e2*np.eye(3)
    #proj_pos = [-20,-10,-20] if in_inertial else [-30,-40,-30]
    #ax_lim_up = [20,25,120] if in_inertial else [20,30,120]
    proj_pos = [-40,-25,-20] if in_inertial else [-40,-80,-40]
    ax_lim_up = [20,25,120] if in_inertial else [20,30,120]
    
    fig = plt.figure(1,figsize=[6,6.8])
    plt.clf()
    ax = fig.add_subplot(111,projection='3d')
    ax.view_init(elev=23,azim=48)
    ax.plot([xtraj[0][0]],[xtraj[1][0]],[xtraj[2][0]],**init3d_style)
    ax.plot(xtraj[0],xtraj[1],xtraj[2],**traj3d_style)
    for k in range(xtraj.shape[1]):
        # Velocity
        vel_vec_0 = xtraj[:3,k]
        vel_vec_1 = xtraj[:3,k]+vel_scale.dot(xtraj[3:,k])
        vel_vec = np.column_stack([vel_vec_0,vel_vec_1])
        ax.plot(vel_vec[0],vel_vec[1],vel_vec[2],**vel3d_style)
    u_total = sum(utraj)
    for k in range(u_total.shape[1]):
        # Acceleration (input), in the opposite direction as if thrust
        accel_vec_0 = xtraj[:3,k]
        accel_vec_1 = xtraj[:3,k]-accel_scale.dot(u_total[:,k])
        accel_vec = np.column_stack([accel_vec_0,accel_vec_1])
        ax.plot(accel_vec[0],accel_vec[1],accel_vec[2],**accel3d_style)
    for i in range(3):
        traj2d = xtraj.copy()
        u_total2d = u_total.copy()
        traj2d[i] = proj_pos[i]
        traj2d[i+3] = 0
        u_total2d[i] = 0
        ax.plot(traj2d[0],traj2d[1],traj2d[2],**traj2d_style)
        ax.plot([traj2d[0][0]],[traj2d[1][0]],[traj2d[2][0]],**init2d_style)
        for k in range(traj2d.shape[1]):
            # Velocity
            vel_vec_0 = traj2d[:3,k]
            vel_vec_1 = traj2d[:3,k]+vel_scale.dot(traj2d[3:,k])
            vel_vec = np.column_stack([vel_vec_0,vel_vec_1])
            ax.plot(vel_vec[0],vel_vec[1],vel_vec[2],**vel2d_style)
        for k in range(u_total.shape[1]):
            # Acceleration (input), in the opposite direction as if thrust
            accel_vec_0 = traj2d[:3,k]
            accel_vec_1 = traj2d[:3,k]-accel_scale.dot(u_total2d[:,k])
            accel_vec = np.column_stack([accel_vec_0,accel_vec_1])
            ax.plot(accel_vec[0],accel_vec[1],accel_vec[2],**accel2d_style)
    ax.set_xlabel('$x$ position [m]',labelpad=10)
    ax.set_ylabel('$y$ position [m]',labelpad=10)
    ax.set_zlabel('$z$ position [m]',labelpad=10)
    ax.set_xlim([proj_pos[0],ax_lim_up[0]])
    ax.set_ylim([proj_pos[1],ax_lim_up[1]])
    ax.set_zlim([proj_pos[2],ax_lim_up[2]])
    # Remove fill
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    # Now set color to white (or whatever is "invisible")
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    plt.tight_layout()
    
    if save_pdf:
        fig.savefig('figures/%s/docking_trajectory_%s.pdf'%(folder,frame),
                    bbox_inches='tight',format='pdf',transparent=True)
    
    #%% Control history
    
    matplotlib.rcParams.update({'font.size': 15})
    
    tu = t[:-1]/t[-1]
    N = tu.size
    
    u_style = dict(edgecolor='black',color='black',alpha=0.3,step='post',linewidth=0,zorder=90)
    sigma_style = dict(linestyle='none',color='blue',marker='.',markersize=2,zorder=91)
    u_scale = 1e3
    
    fig = plt.figure(2,figsize=(5.8,8.2))
    plt.clf()
    ax = fig.add_subplot(511)
    u_ = [np.column_stack([utraj[i],utraj[i][:,-1]]) for i in range(M)]
    sigma_ = [np.append(primal['sigma'][i],primal['sigma'][i][-1]) for i in range(M)]
    sum_u_total = sum([la.norm(u_[i]*u_scale,axis=0) for i in range(len(u_))])
    sum_sigma = sum(sigma_)*u_scale
    ax.fill_between(t/t[-1],sum_u_total,**u_style)
    ax.plot(t/t[-1],sum_sigma,**sigma_style)
    for i in range(K):
        ax.axhline(rho2*(i+1)*u_scale,linewidth=1,linestyle='--',color='gray')
    ax.axhline(rho1*u_scale,linewidth=1,linestyle='--',color='gray')
    ax.autoscale(tight=True)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('$\sum_{i=1}^M\|u_i(t)\|_2$')
    ax.set_yticks(ticks=[i*rho2*u_scale for i in range(K+1)])
    ax.set_xticks(ticks=np.linspace(0,1,3))
    ax.set_xlim([0,1])
    for i in range(M):
        ax = fig.add_subplot(5,3,4+i)
        ax.fill_between(t/t[-1],la.norm(u_[i],axis=0)*u_scale,**u_style)
        ax.plot(t/t[-1],sigma_[i]*u_scale,**sigma_style)
        ax.axhline(rho2*u_scale,linewidth=1,linestyle='--',color='gray')
        ax.axhline(rho1*u_scale,linewidth=1,linestyle='--',color='gray')
        ax.autoscale(tight=True)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('$\|u_{%d}(t)\|_2$'%(i+1))
        ax.set_xticks(ticks=np.linspace(0,1,3))
        ax.set_xlim([0,1])
    plt.tight_layout(pad=0.5,h_pad=0,w_pad=0)
    
    if save_pdf:
        fig.savefig('figures/%s/docking_input.pdf'%(folder),
                    bbox_inches='tight',format='pdf',transparent=True)
    
    #%% Thruster layout
    
    matplotlib.rcParams.update({'font.size': 18})
    
    u_style = dict(linewidth=2,zorder=99)
    u2d_style = dict(linewidth=1.5,alpha=0.5,zorder=90)
    offset = 1.2
    proj_pos = [-2,-2,-1.5]
    
    # Colors
    u_color = sns.color_palette(None,len(C))
    
    fig = plt.figure(3,figsize=[5.3,5.7])
    plt.clf()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=23,azim=40)
    nhat = [-C[i][-1] for i in range(len(C))] # Cone pointing directions
    for i in range(len(C)):
        ax.plot([0,nhat[i][0]],[0,nhat[i][1]],[0,nhat[i][2]],color=u_color[i],**u_style)
        ax.text(nhat[i][0]*offset,nhat[i][1]*offset,nhat[i][2]*offset,'$u_{%d}$'%(i+1),color=u_color[i])
        for j in range(3):
            nhat2d = nhat[i].copy()
            nhat2d[j] = proj_pos[j]
            nhat_vec_0 = np.array([0 if k!=j else proj_pos[j] for k in range(3)])
            nhat_vec_1 = nhat2d
            nhat_vec = np.column_stack([nhat_vec_0,nhat_vec_1])
            ax.plot(nhat_vec[0],nhat_vec[1],nhat_vec[2],color=u_color[i],**u2d_style)
    ax.set_xlabel('$u_{i,1}$',labelpad=0)
    ax.set_ylabel('$u_{i,2}$',labelpad=0)
    ax.set_zlabel('$u_{i,3}$',labelpad=0)
    ax.set_xlim([proj_pos[0],1.5])
    ax.set_ylim([proj_pos[1],1.5])
    ax.set_zlim([proj_pos[2],1.5])
    ax.set_xticks([proj_pos[0],1.5])
    ax.set_yticks([proj_pos[1],1.5])
    ax.set_zticks([proj_pos[2],1.5])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    # Remove fill
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    # Now set color to white (or whatever is "invisible")
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    plt.tight_layout()
    
    if save_pdf:
        fig.savefig('figures/%s/docking_rcs.pdf'%(folder),
                    bbox_inches='tight',format='pdf',transparent=True)
        
    #%% Iput gain values
    
    u_on_tol = 1e-5
    colors = plt.get_cmap('tab10')
    
    nhat = [-C[i][-1] for i in range(M)]
    Gamma = [np.array([nhat[i].dot(misc['y'][k]) for k in range(N)]) for i in range(M)]
    Gamma_on = [np.array([Gamma[i][k] if la.norm(primal['u'][i][:,k])>u_on_tol else np.nan for k in range(N)]) for i in range(M)]
    Gamma_off = [np.array([Gamma[i][k] if la.norm(primal['u'][i][:,k])<=u_on_tol else np.nan for k in range(N)]) for i in range(M)]
    
    fig = plt.figure(4,figsize=(5.8,8.2))
    plt.clf()
    ax = fig.add_subplot(111)
    for i in range(M):
        ax.plot(tu,Gamma_on[i],color=colors(i),label='$\Gamma_{%d}(t)$'%(i+1))
        ax.plot(tu,Gamma_on[i],linewidth=3,color=colors(i))
        ax.plot(tu,Gamma_off[i],linestyle=':',color=colors(i))
    ax.axhline(0,color='red',linestyle='--')
    ax.autoscale(tight=True)
    ax.set_xlabel('Normalized time')
    ax.set_ylabel('$\Gamma_i(t)$')
    plt.tight_layout()
    plt.subplots_adjust(top=0.91,bottom=0.087)
    ax.legend(loc='right',ncol=6,labelspacing=0.1,columnspacing=0.7,
              handlelength=1,prop={'size':12},bbox_to_anchor=(1.02,1.05))
    
    if save_pdf:
        fig.savefig('figures/%s/input_gain.pdf'%(folder),
                        bbox_inches='tight',format='pdf',transparent=True)
