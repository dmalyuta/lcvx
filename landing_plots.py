"""
Plots for the paper for docking.

D. Malyuta -- ACL, University of Washington
B. Acikmese -- ACL, University of Washington

Copyright 2019 University of Washington. All rights reserved.
"""

import os
import pickle
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import matplotlib

import tools

def plot_ifac20(data,save_pdf=False,folder=''):
    matplotlib.rc('font',**{'family':'serif','serif':['DejaVu Sans']})
    matplotlib.rc('text', usetex=True)
    
    #%% User choices
    
    save_pdf = save_pdf # True to save figure PDFs
    legend_fontsize = 13 # Size of the legend font
    
    #%% Load data
    
    solution = pickle.load(open(data,'rb'))
    t,primal,dual,misc = solution['t'],solution['primal'],solution['dual'],solution['misc']
    
    print 'Time of flight %.2f [s]'%(t[-1])

    M,rho1,rho2,theta,gs,C,zeta,Ex = (solution['M'],solution['rho1'],
                                      solution['rho2'],solution['theta'],
                                      solution['gs'],solution['C'],
                                      solution['zeta'],solution['Ex'])
    
    if save_pdf and not os.path.isdir('figures/%s'%(folder)):
        os.makedirs('figures/%s'%(folder))
    
    #%% Helper functions
    
    def post_process(primal):
        """Remove same-direction thrust artifacts."""
        alignment_tol = np.cos(np.deg2rad(1)) # [deg] Input alignment tolerance
        mag_tol = 1e-3 # [m/s^2] Input magnitude tolerance
        N = primal['x'].shape[1]-1 # Number of input grid points
        for k in range(N-1):
            inputs = [primal['u'][i][:,k] for i in range(2)]
            inputs_next = [primal['u'][i][:,k+1] for i in range(2)]
            alignment = inputs[0].dot(inputs[1])/(la.norm(inputs[0])*la.norm(inputs[1]))
            if alignment>=alignment_tol and la.norm(inputs[0])>mag_tol and la.norm(inputs[1])>mag_tol:
                j = 0 if la.norm(inputs_next[0])>la.norm(inputs_next[1]) else 1
                for field in ['u','gamma','sigma']:
                    if field=='u':
                        primal[field][j][:,k] = np.sum([primal[field][i][:,k] for i in range(2)],axis=0)
                        primal[field][1-j][:,k] = np.zeros(2)
                    else:
                        primal[field][j][k] = np.sum([primal[field][i][k] for i in range(2)],axis=0)
                        primal[field][1-j][k] = 0
        return primal
    
    def make_arc(angle,magnitude,direction):
        """
        Make an arc for a polar plot.
        
        Parameters
        ----------
        angle : float
            The half-angle of the cone that the arc spans.
        magnitude : float
            The distance of the arc from the origin.
        direction : array
            The direction of the cone that the arc spans.
        """
        n = direction*magnitude
        angle = np.deg2rad(angle)
        angles = np.linspace(-angle/2,angle/2)
        arc = np.zeros((2,angles.size))
        for i in range(angles.size):
            arc[:,i] = tools.R2d(angles[i]).dot(n)
        return arc
    
    def draw_polar_grid(ax,angle,bounds,direction,angle_ticks,magnitude_ticks,
                        border_style,grid_style,labels=True,angle_tick_blacklist=None):
        """
        Draw a polar grid.
        
        Parameters
        ----------
        ax : AxesSubplot
            Axes handle on which to plot.
        angle : float
            Polar plot half-angle.
        bounds : 2-element tuple
            The (lower,upper) magnitude bounds of the polar plot.
        direction : array
            Polar plot "nominal" (zero angle) direction.
        """
        eps = np.finfo(np.float64).eps
        ang_text_style = dict(va='bottom',ha='center',fontsize=10,color=0.2*np.ones(3))
        mag_text_style = dict(va='center',ha='right',fontsize=10,color=0.2*np.ones(3))
        mag_offset = 0.5
        ang_offset = 0.4
        # Draw the polar plot border
        arc1 = make_arc(angle,bounds[0],direction)
        arc2 = np.fliplr(make_arc(angle,bounds[1],direction))
        arc = np.column_stack([arc1,arc2,arc1[:,0]])
        ax.plot(arc[0],arc[1],**border_style)
        if labels:
            arc_ = make_arc(angle,bounds[1]+mag_offset,direction)
            ax.text(arc_[0][-1],arc_[1][-1],'$%d^\circ$'%(angle/2),**ang_text_style)
            ax.text(arc_[0][0],arc_[1][0],'$%d^\circ$'%(-angle/2),**ang_text_style)
        # Draw the polar plot grid
        for ang in angle_ticks:
            if np.abs(ang)<angle/2-eps:
                grid_line = np.column_stack([bnd*tools.R2d(np.deg2rad(ang)).dot(direction) for bnd in bounds])
                ax.plot(grid_line[0],grid_line[1],**grid_style)
                if labels and not angle_tick_blacklist(ang):
                    pos_ = np.column_stack([(bnd+mag_offset)*tools.R2d(np.deg2rad(ang)).dot(direction) for bnd in bounds])
                    ax.text(pos_[0][-1],pos_[1][-1],'$%d^\circ$'%(ang),**ang_text_style)
        for mag in magnitude_ticks:
            if mag>=bounds[0] and mag<=bounds[1]:
                grid_line = make_arc(angle,mag,direction)
                if mag>bounds[0]+eps and mag<bounds[1]-eps:
                    ax.plot(grid_line[0],grid_line[1],**grid_style)
                if labels:
                    pos_ = make_arc(angle+np.rad2deg(ang_offset/mag)*2,mag,direction)
                    ax.text(pos_[0][-1],pos_[1][-1],'$%d$~m/s$^2$'%(mag),**mag_text_style)
    
    primal = post_process(primal)
    
    colors = ['green','blue','red','cyan','magenta']
    
    #### Trajectory plot
    
    tu = t[:-1]/t[-1]
    dt = t[1]-t[0]
    N = tu.size
    
    pos_style = dict(color='black',linewidth=1,zorder=90)
    thrust_style = [dict(color=colors[i],linewidth=0.5) for i in range(M)]
    ground_style = dict(linewidth=0,color='black',alpha=0.2,zorder=1)
    gs_style = dict(color='black',linestyle='--',linewidth=0.5,zorder=1)
    state_constraint_active_style = dict(color='red',marker='.',linestyle='none',markersize=10,zorder=99)
    u_whisker_scale = 20
    
    active_tol = 1e-4#np.sqrt(np.finfo(np.float64).eps)
    state_constraint = lambda x: (np.array([0,1]).dot(Ex).dot(primal['x'][:,k])-
                                  la.norm(Ex.dot(primal['x'][:,k]))*np.cos(np.pi/2-gs))<=active_tol
    try:
        state_constraint_active = np.column_stack([primal['x'][:,k] for k in range(N) if state_constraint(primal['x'][:,k])])
    except ValueError:
        state_constraint_active = None
    
    matplotlib.rcParams.update({'font.size': 26})
    
    n_gs = [tools.R2d(-(np.pi/2-gs)).dot(np.array([0,1])),
            tools.R2d(np.pi/2-gs).dot(np.array([0,1]))]
    
    fig = plt.figure(1,figsize=(9,4))
    plt.clf()
    ax = fig.add_subplot(111)
    ax.axis('equal')
    ax.plot(primal['x'][0],primal['x'][1],**pos_style)
    for k in range(primal['x'].shape[1]-1):
        for i in range(M):
            ax.plot([primal['x'][0,k],primal['x'][0,k]-u_whisker_scale*primal['u'][i][0,k]],
                    [primal['x'][1,k],primal['x'][1,k]-u_whisker_scale*primal['u'][i][1,k]],**thrust_style[i])
    if state_constraint_active is not None:
        ax.plot(state_constraint_active[0],state_constraint_active[1],label='$x(t)\in\mathrm{bdry}(\mathcal X)$',**state_constraint_active_style)
    ax.set_xlabel('Downrange [m]',fontsize=22)
    ax.set_ylabel('Altitude AGL [m]',fontsize=22)
    fig.tight_layout()
    ax.autoscale(tight=True)
    fig.canvas.draw()
    xlims = ax.get_xlim()
    ylims = ax.get_ylim()
    ax.fill_between(xlims,[ylims[0],ylims[0]],[0,0],**ground_style)
    s = ylims[1]/n_gs[0][1]
    for i in range(2):
        ax.plot([0,s*n_gs[i][0]],[0,s*n_gs[i][1]],**gs_style)
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.set_yticks(ticks=np.linspace(0,primal['x'][1,0],3))
    
    if save_pdf:
        fig.savefig('figures/%s/rocket_trajectory.pdf'%(folder),bbox_inches='tight',format='pdf',transparent=True)
    
    #### Thrust magnitude plot
    
    matplotlib.rcParams.update({'font.size': 20})
    
    border_style = [dict(color=colors[i],linewidth=1.5) for i in range(M)]
    grid_style = [dict(color=colors[i],linewidth=0.5,alpha=0.3) for i in range(M)]
    thrust_style = [dict(color=colors[i],marker='x',markersize=5,linestyle='none',zorder=90) for i in range(M)]
    origin_style = dict(color='black',linestyle='none',marker='.',markersize=5,zorder=100)
    angles = np.linspace(-90.,90.,np.int(180./20.+1))
        
    u_style = dict(edgecolor='black',color='black',alpha=0.3,step='post',linewidth=0,zorder=90)
    sigma_style = dict(linestyle='none',color='blue',marker='.',markersize=2,zorder=91)
    u_scale = 1
    
    fig = plt.figure(3,figsize=[7.5,5])
    plt.clf()
    ax = fig.add_subplot(211)
    u_ = [np.column_stack([primal['u'][i],primal['u'][i][:,-1]]) for i in range(M)]
    sigma_ = [np.append(primal['sigma'][i],primal['sigma'][i][-1]) for i in range(M)]
    sum_u_total = sum([la.norm(u_[i]*u_scale,axis=0) for i in range(len(u_))])
    sum_sigma = sum(sigma_)*u_scale
    ax.fill_between(t/t[-1],sum_u_total,**u_style)
    ax.plot(t/t[-1],sum_sigma,**sigma_style)
    for i in range(M):
        ax.axhline(rho2[i]*u_scale,linewidth=1,linestyle='--',color='gray')
    ax.autoscale(tight=True)
    ax.set_xlabel('Normalized time $\hat t$',fontsize=16)
    ax.set_ylabel('$\sum_{i=1}^M\|u_i(t)\|_2$ [m/s$^2$]',fontsize=16)
    ax.set_yticks(ticks=np.concatenate([[0],[rho2[i]*u_scale for i in range(M)]]))
    ax.set_xticks(ticks=np.linspace(0,1,5))
    ax.set_xlim([0,1])
    for i in range(M):
        ax = fig.add_subplot(2,M,M+1+i)
        ax.fill_between(t/t[-1],la.norm(u_[i],axis=0)*u_scale,**u_style)
        ax.plot(t/t[-1],sigma_[i]*u_scale,**sigma_style)
        ax.axhline(rho2[i]*u_scale,linewidth=1,linestyle='--',color='gray')
        ax.axhline(rho1[i]*u_scale,linewidth=1,linestyle='--',color='gray')
        ax.autoscale(tight=True)
        ax.set_xlabel('Normalized time $\hat t$',fontsize=16)
        ax.set_ylabel('$\|u_{%d}(t)\|_2$ [m/s$^2$]'%(i+1),fontsize=16)
        ax.set_yticks(ticks=np.linspace(0,rho2[i],5))
        ax.set_xticks(ticks=np.linspace(0,1,5))
        ax.set_xlim([0,1])
    plt.tight_layout(pad=0.5,h_pad=0,w_pad=0.5)
    
    if save_pdf:
        fig.savefig('figures/%s/rocket_input_magnitude.pdf'%(folder),bbox_inches='tight',format='pdf',transparent=True)
        
    if 'micp' in data:
        return # Dual variables don't apply to MICP solution
    
    #### DUAL VARIABLE PLOTS
    
    matplotlib.rcParams.update({'font.size': 16})
        
    #### Input valuation plot (support functions)
    
    projection = [np.array([la.norm(misc['y'][k]-C[i].T.dot(dual['lambda_4_%d'%(i+1)][:,k]),ord=2)
                            for k in range(N)]) for i in range(M)]
    gain = [np.array([(projection[i][k]+dual['eta'][0,k]*dt*zeta)*rho2[i]
            for k in range(N)]) for i in range(M)]
    dgain = gain[1]-gain[0]
    ndgain = np.array([1/dgain[k] if np.abs(dgain[k])>1e-6 else np.nan for k in range(N)])#/min(np.max(dgain),-np.min(dgain))
    proj_tol = np.sqrt(np.finfo(np.float64).eps)
    tu_ndgain_zero = np.array([tu[k] for k in range(N) if np.isnan(ndgain[k]) and not np.any([
            projection[i][k]<proj_tol for i in range(M)])])
    
    u_active_tol = np.sqrt(1e-5)
    u_active = [np.array([la.norm(primal['u'][i][:,k])>u_active_tol for k in range(N)]) for i in range(M)]
    u_active_style = [dict(color=colors[i],alpha=0.2,step='post',linewidth=0,zorder=0) for i in range(M)]
    
    y2_color = 'magenta'
    
    fig = plt.figure(4,figsize=(6.4,3.5))
    plt.clf()
    ax = fig.add_subplot(111)
    for i in range(M):
        ax.plot(tu,gain[i],color=colors[i],label='$\Gamma_{%d}(t)$'%(i+1))
    ax.axhline(0,color='red',linewidth=1,linestyle='--',zorder=1)
    ax.autoscale(tight=True)
    ax.set_ylabel('Gain $\Gamma_i(t)$')
    ax.set_xlabel('Normalized time')
    ax.legend(prop={'size': legend_fontsize})
    fig.canvas.draw()
    xlims = ax.get_xlim() 
    ylims = ax.get_ylim()
    for i in range(M):
        ax.fill_between(tu,u_active[i]*ylims[1],u_active[i]*ylims[0],**u_active_style[i])
    ax2 = ax.twinx()
    ax.spines['right'].set_color(y2_color)
    ax2.spines['right'].set_color(y2_color)
    ax2.tick_params(axis='y', colors=y2_color)
    ax2.set_yscale('symlog')
    ax2.plot(tu,ndgain,color=y2_color)
    for k in range(tu_ndgain_zero.size):    
        ax2.axvline(tu_ndgain_zero[k],color='black',linestyle='--')
    ax2.axhline(0,color='red',linewidth=1,linestyle='--',zorder=1)
    ax2.autoscale(tight=True)
    ax2.set_ylabel('$1/(\Gamma_2(t)-\Gamma_1(t))$',color=y2_color)
    ax.set_zorder(ax2.get_zorder()+1)
    ax.patch.set_visible(False) 
    plt.tight_layout()
    
    if save_pdf:
        fig.savefig('figures/%s/rocket_input_valuation.pdf'%(folder),bbox_inches='tight',format='pdf',transparent=True)
        
    #### Primer vector trajectory
    
    if zeta==0:
        equiv_plane_angle = np.pi/2-np.deg2rad(theta[1]/2)-np.arccos(np.float(rho2[0])/rho2[1])
        n_equiv = np.array([np.cos(equiv_plane_angle),np.sin(equiv_plane_angle)])
    
    try:
        state_constraint_active = np.column_stack([misc['y'][k,:] for k in range(N) if state_constraint(primal['x'][:,k])])
    except ValueError:
        state_constraint_active = None
    
    primer_style = dict(color='black',zorder=999)
    primer_style_ic = dict(color='black',marker='x',linestyle='none',zorder=999)
    equiv_plane_style = gs_style
    yscale = rho2[0]/np.max(la.norm(misc['y'],axis=1))
    
    fig = plt.figure(6,figsize=(6.4,4.8))
    plt.clf()
    ax = fig.add_subplot(111)
    ax.axis('equal')
    for i in range(M):
        draw_polar_grid(ax,theta[i],[rho1[i],rho2[i]],np.array([0,1]),
                        angles,np.array(range(rho1[i]+i,rho2[i]+1)),
                        border_style[i],grid_style[i],
                        angle_tick_blacklist=lambda ang:ang<=theta[1]/2 and ang>=-theta[1]/2)
        ax.plot([u if la.norm(u)>u_active_tol else np.nan for u in primal['u'][i][0]],
                [u if la.norm(u)>u_active_tol else np.nan for u in primal['u'][i][1]],
                label='$u_{%d}(t)$'%(i),**thrust_style[i])
    ax.plot(0,0,**origin_style)
    ax.plot(yscale*misc['y'][:,0],yscale*misc['y'][:,1],label='$\hat y(t)$',**primer_style)
    ax.plot(yscale*misc['y'][0,0],yscale*misc['y'][0,1],label='$\hat y(0)$',**primer_style_ic)
    if state_constraint_active is not None:
        ax.plot(yscale*state_constraint_active[0],yscale*state_constraint_active[1],label='$x(t)\in\mathrm{bdry}(\mathcal X)$',**state_constraint_active_style)
    ax.set_xlabel('$x$-acceleration $u_{i,1}(t)$ [m/s$^2$]',fontsize=16)
    ax.set_ylabel('$y$-acceleration $u_{i,2}(t)$ [m/s$^2$]',fontsize=16)
    plt.tight_layout()
    ylims = ax.get_ylim()
    ylims = [-1,14]
    ax.set_ylim(ylims)
    fig.canvas.draw()
    xlims = ax.get_xlim()
    if zeta==0:
        s = xlims[1]*2/n_equiv[0]
        ax.plot([0,s*n_equiv[0]],[0,s*n_equiv[1]],**equiv_plane_style)
        ax.plot([0,-s*n_equiv[0]],[0,s*n_equiv[1]],**equiv_plane_style)
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.legend(loc='upper right',prop={'size': legend_fontsize})
    
    if save_pdf:
        fig.savefig('figures/%s/rocket_primer_vector.pdf'%(folder),bbox_inches='tight',format='pdf',transparent=True)
        
def plot_ifac2020_sweep(h0,mintime,save_pdf=False):
    """
    Plot a sweep of landing trajectories over the initial altitude AGL.
    
    Parameters
    ----------
    h0 : np.array
        The initial altitudes AGL at which to compute trajectories.
    mintime : bool
        True for minimum-time, False for minimum-fuel.
    save_pdf : bool, optional
        True to save a PDF of the figure.
    """
    import lcvx
    import landing_2mode
    
    matplotlib.rc('font',**{'family':'serif','serif':['DejaVu Sans']})
    matplotlib.rc('text', usetex=True)
    matplotlib.rcParams.update({'font.size': 15})
    
    pos_style = dict(linewidth=1)
    ground_style = dict(linewidth=0,color='black',alpha=0.2,zorder=1)
    gs_style = dict(color='black',linestyle='--',linewidth=0.5,zorder=1)
    
    fig = plt.figure(20,figsize=(5,11.45))
    plt.clf()
    ax = fig.add_subplot(111)
    #ax.axis('equal')
    total_solver_time = 0
    for i in range(h0.size):
        print '\n======== Running (%d/%d): h0 = %.1f m AGL'%(i+1,h0.size,h0[i])
        frac = np.float(i)/(h0.size-1)
        rocket = landing_2mode.Lander(agl=h0[i],N=30,mintime=mintime)
        J,t,primal,dual,misc,solver_time = lcvx.solve(rocket,[0.,100.],opt_tol=1e-4)
        total_solver_time += solver_time
        ax.plot(primal['x'][0],primal['x'][1],color=(1-frac)*np.zeros(3)+frac*np.ones(3)*0.7,zorder=1000-i,**pos_style)
        if i==0:
            n_gs = [tools.R2d(-(np.pi/2-rocket.gs)).dot(np.array([0,1])),
                    tools.R2d(np.pi/2-rocket.gs).dot(np.array([0,1]))]
    print '\nTotal solver time: %f s'%(total_solver_time)
    ax.set_xlabel('Downrange [m]',fontsize=15)
    ax.set_ylabel('Altitude AGL [m]',fontsize=15)
    fig.tight_layout()
    ax.autoscale(tight=True)
    fig.canvas.draw()
    xlims = ax.get_xlim()
    ylims = ax.get_ylim()
    ax.fill_between(xlims,[ylims[0],ylims[0]],[0,0],**ground_style)
    s = ylims[1]/n_gs[0][1]
    for i in range(2):
        ax.plot([0,s*n_gs[i][0]],[0,s*n_gs[i][1]],**gs_style)
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.set_yticks(ticks=np.linspace(0,primal['x'][1,0],7))
    
    if save_pdf:
        fig.savefig('figures/sweep_h0_%d_%d_zeta%d.pdf'%(h0[0],h0[-1],0 if mintime else 1),
                    bbox_inches='tight',format='pdf',transparent=True)
