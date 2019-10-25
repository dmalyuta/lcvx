"""
2D rocket landing with a thruster that can produce either low thrusts with a
large gimbal angle, or large thrusts with a small gimbal angle.

D. Malyuta -- ACL, University of Washington
B. Acikmese -- ACL, University of Washington

Copyright 2019 University of Washington. All rights reserved.
"""

import numpy as np
import numpy.linalg as la
import cvxpy as cvx

import lcvx
import tools

#%% Problem definition

class Lander(lcvx.Problem,object):
    def __init__(self,micp=False):
        """
        Parameters
        ----------
        micp : bool, optional
            Set to ``True`` to solve the problem via mixed-integer programming.
        """
        super(Lander, self).__init__()
        
        cvx_opts = dict(solver=cvx.GUROBI,verbose=False,Presolve=1,LogFile='',threads=1)
        #cvx_opts = dict(solver=cvx.ECOS,verbose=False)
        
        # Physical parameters
        self.omega = 2*np.pi/(24*3600+39*60+35) # [rad/s] Mars spin
        Tmax = 21500. # [N] Maximum thrust
        self.mass = 1700. # [kg] Lander mass
        amax = Tmax/self.mass; # [m/s^2] Maximum acceleration
        self.g = 3.71 # [m/s^2] Mars surface gravity
        self.R = 3396.2e3 # [m] Mars radius
        self.rho1 = [4,8]#[s_*amax for s_ in [0.2,0.6]] # [m/s^2] Smallest  acceleration
        self.rho2 = [8,12]#[s_*amax for s_ in [0.6,0.9]] # [m/s^2] Largest control acceleration
        self.gs = np.deg2rad(85) # [rad] Smallest glideslope angle
        
        # Boundary conditions
        r0 = np.array([-100.,3000.])
        v0 = np.array([-38.,-60.])
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
        scaling = 1 # This affects the magnitude of the primer vector, which improves plotting
        Dx = np.concatenate(np.abs([r0,v0]))
        Dx[Dx==0] = 1
        Dx[0] = r0[1]*np.tan(np.pi/2-self.gs)
        self.Dx = scaling*np.diag(Dx)
        self.Du = [scaling*np.diag([rho2 for _ in range(nu)]) for rho2 in self.rho2]
        self.tfmax = 100.
        
        # Optimization problem common parts
        self.N = 30 # Temporal solution
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
        wx = 1e-3*scaling
        time_penalty = tf/self.tfmax
        input_penalty = xi[-1]/ximax
#        state_penalty = sum([cvx.quad_form(rdt*x[k],Dxi.dot(Q).dot(Dxi)) for k in range(self.N)])
        state_penalty = sum([cvx.abs(dt*Dxi[0,0]*x[k][0])+cvx.abs(dt*Dxi[1,1]*x[k][1])
                             for k in range(self.N)])
        
        self.zeta = 1 # minimum time: 0
        cost_p2 = input_penalty+wx*state_penalty
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
        add_constraint([xi[k+1] == xi[k]+dt*sum([sigma[i][k] for i in range(self.M)]) for k in range(self.N)],'eta')
        x[0].value = np.concatenate([r0,v0])
        xi[0].value = 0
        add_constraint([x[-1] == np.concatenate([rf,vf])],'nu_xN')
        for i in range(self.M):
            add_constraint([cvx.norm2(u[i][k]) <= sigma[i][k] for k in range(self.N)],'beta_1_%d'%(i+1))
            add_constraint([gamma[i][k]*self.rho1[i]-sigma[i][k] <=0 for k in range(self.N)],'beta_2_%d'%(i+1))
            add_constraint([sigma[i][k]-gamma[i][k]*self.rho2[i] <= 0 for k in range(self.N)],'beta_3_%d'%(i+1))
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
        
# Lossless convexification solution
rocket = Lander()
#conditions_hold,info = lcvx.check_conditions_123(rocket)
J,t,primal,dual,misc,solver_time = lcvx.solve(rocket,[0.,200.],opt_tol=1e-4)

#%% Plots

import matplotlib
import matplotlib.pyplot as plt

matplotlib.rc('font',**{'family':'serif','serif':['DejaVu Sans']})
matplotlib.rc('text', usetex=True)

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
                    border_style,grid_style,labels=True):
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
    text_style = dict(va='top')
    # Draw the polar plot border
    arc1 = make_arc(angle,bounds[0],direction)
    arc2 = np.fliplr(make_arc(angle,bounds[1],direction))
    arc = np.column_stack([arc1,arc2,arc1[:,0]])
    ax.plot(arc[0],arc[1],**border_style)
    if labels:
        ax.text(arc2[0][0],arc2[1][0],'$%d^\circ$'%(angle/2),va='top')
        ax.text(arc2[0][-1],arc2[1][-1],
                '$%d^\circ$'%(-angle/2),**text_style)
    # Draw the polar plot grid
    for ang in angle_ticks:
        if np.abs(ang)<angle/2-eps:
            grid_line = np.column_stack([bnd*tools.R2d(np.deg2rad(ang)).dot(direction) for bnd in bounds])
            ax.plot(grid_line[0],grid_line[1],**grid_style)
            if labels:
                ax.text(grid_line[0][-1],grid_line[1][-1],'$%d^\circ$'%(ang),**text_style)
    for mag in magnitude_ticks:
        if mag>bounds[0]+eps and mag<bounds[1]-eps:
            grid_line = make_arc(angle,mag,direction)
            ax.plot(grid_line[0],grid_line[1],**grid_style)
            if labels:
                ax.text(grid_line[0][-1],grid_line[1][-1],'$%d$~m/s$^2$'%(mag),**text_style)

primal = post_process(primal)

save_pdf = False # True to save figure PDFs

K = rocket.K
M = rocket.M
rho2 = rocket.rho2
rho1 = rocket.rho1
theta = rocket.theta
gs = rocket.gs
C = rocket.C

colors = ['green','blue','red','cyan','magenta']

#### Trajectory plot

pos_style = dict(color='black',linewidth=0.5,zorder=90)
vel_style = dict(color='red',linewidth=0.5)
thrust_style = [dict(color=colors[i]) for i in range(M)]
ground_style = dict(linewidth=0,color='black',alpha=0.2,zorder=1)
gs_style = dict(color='black',linestyle='--',linewidth=0.5,zorder=1)
v_whisker_scale = 5
u_whisker_scale = 20

matplotlib.rcParams.update({'font.size': 18})

n_gs = [tools.R2d(-(np.pi/2-gs)).dot(np.array([0,1])),
        tools.R2d(np.pi/2-gs).dot(np.array([0,1]))]

fig = plt.figure(1,figsize=(4.5,8.5))
plt.clf()
ax = fig.add_subplot(111)
ax.axis('equal')
ax.plot(primal['x'][0],primal['x'][1],**pos_style)
#for k in range(primal['x'].shape[1]):
#    ax.plot([primal['x'][0,k],primal['x'][0,k]+v_whisker_scale*primal['x'][2,k]],
#            [primal['x'][1,k],primal['x'][1,k]+v_whisker_scale*primal['x'][3,k]],**vel_style)
for k in range(primal['x'].shape[1]-1):
    for i in range(M):
        ax.plot([primal['x'][0,k],primal['x'][0,k]-u_whisker_scale*primal['u'][i][0,k]],
                [primal['x'][1,k],primal['x'][1,k]-u_whisker_scale*primal['u'][i][1,k]],**thrust_style[i])
ax.set_xlabel('$x$ position [m]')
ax.set_ylabel('$y$ position [m]')
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

#### Thrust polar plot

border_style = [dict(color=colors[i],linewidth=1.5) for i in range(M)]
grid_style = [dict(color=colors[i],linewidth=0.5,alpha=0.3) for i in range(M)]
thrust_style = [dict(color=colors[i],marker='s',markersize=5,linestyle='none',zorder=90) for i in range(M)]
origin_style = dict(color='black',linestyle='none',marker='.',markersize=5,zorder=100)
angles = np.linspace(-90.,90.,np.int(180./20.+1))
magnitudes = np.linspace(0,np.ceil(rho2[-1]),np.int(np.ceil(rho2[-1])/1.+1))
thrust_direction = np.array([0,-1])

if save_pdf:
    fig.savefig('figures/rocket_trajectory.pdf',bbox_inches='tight',format='pdf',transparent=True)

matplotlib.rcParams.update({'font.size': 14})

fig = plt.figure(2,figsize=(4,5))
plt.clf()
ax = fig.add_subplot(111)
ax.axis('equal')
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
plt.tight_layout()
plt.axis('off')
ax.plot(0,0,**origin_style)
for i in range(M):
    draw_polar_grid(ax,theta[i],[rho1[i],rho2[i]],thrust_direction,angles,
                    magnitudes,border_style[i],grid_style[i])
    ax.plot([-u if la.norm(u)>np.sqrt(1e-5) else np.nan for u in primal['u'][i][0]],
            [-u if la.norm(u)>np.sqrt(1e-5) else np.nan for u in primal['u'][i][1]],**thrust_style[i])
    
if save_pdf:
    fig.savefig('figures/rocket_input_polar.pdf',bbox_inches='tight',format='pdf',transparent=True)

#### Thrust magnitude plot

u_style = dict(edgecolor='black',color='black',alpha=0.3,step='post',zorder=90)
sigma_style = dict(linestyle='none',color='blue',marker='.',markersize=2,zorder=91)
u_scale = 1

fig = plt.figure(3,figsize=[7.5,5])
plt.clf()
ax = fig.add_subplot(211)
sum_u_total = sum([la.norm(primal['u'][i]*u_scale,axis=0) for i in range(len(primal['u']))])
sum_sigma = sum(primal['sigma'])*u_scale
ax.fill_between(t[:-1],sum_u_total,**u_style)
ax.plot(t[:-1],sum_sigma,**sigma_style)
for i in range(M):
    ax.axhline(rho2[i]*u_scale,linewidth=1,linestyle='--',color='gray')
ax.autoscale(tight=True)
ax.set_xlabel('Time [s]')
ax.set_ylabel('$\sum_{i=1}^M\|u_i(t)\|_2$')
ax.set_yticks(ticks=np.concatenate([[0],[rho2[i]*u_scale for i in range(M)]]))
ax.set_xticks(ticks=np.linspace(0,t[-2],5))
ax.set_xlim([0,t[-2]])
for i in range(M):
    ax = fig.add_subplot(2,M,M+1+i)
    ax.fill_between(t[:-1],la.norm(primal['u'][i],axis=0)*u_scale,**u_style)
    ax.plot(t[:-1],primal['sigma'][i]*u_scale,**sigma_style)
    ax.axhline(rho2[i]*u_scale,linewidth=1,linestyle='--',color='gray')
    ax.axhline(rho1[i]*u_scale,linewidth=1,linestyle='--',color='gray')
    ax.autoscale(tight=True)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('$\|u_{%d}(t)\|_2$'%(i+1))
    ax.set_yticks(ticks=np.linspace(0,rho2[i],5))
    ax.set_xticks(ticks=np.linspace(0,t[-2],3))
    ax.set_xlim([0,t[-2]])
plt.tight_layout(pad=0.5,h_pad=0,w_pad=0)

if save_pdf:
    fig.savefig('figures/rocket_input_magnitude.pdf',bbox_inches='tight',format='pdf',transparent=True)
    
#### Input valuability plot (support functions)
    
#%%

cost_coeff = [-(dual['beta_2_%d'%(i+1)]-dual['beta_3_%d'%(i+1)]).flatten() for i in range(M)]
cost_coeff /= np.max(np.abs(cost_coeff))
input_active = [la.norm(primal['u'][i],axis=0)>1e-3 for i in range(M)]
y = misc['y']
valuation = [[tools.support(y[k],C[i])*rho2[i] for k in range(y.shape[0])] for i in range(M)]
valuation /= np.max(np.abs(valuation)) # normalize
r_ = rho2[1]/rho2[0]
a_ = np.cos(np.deg2rad(theta[0]-theta[1])/2)
b_ = np.sin(np.deg2rad(theta[0]-theta[1])/2)
equiv_plane_angle = np.pi/2-np.deg2rad(theta[0]/2)-np.arccos((((r_*a_-1)/(r_*b_))**2+1)**-0.5)
n_equiv = np.array([np.cos(equiv_plane_angle),np.sin(equiv_plane_angle)])

u_active_style = dict(linewidth=0,alpha=0.15,step='post',zorder=1)
border_style = [dict(color=colors[i],linewidth=0.5) for i in range(M)]
dual_style = dict(marker='.',markersize=5)

fig = plt.figure(4,figsize=(6.4,4.8))
plt.clf()
ax = fig.add_subplot(111)
for i in range(M):
    ax.plot(t[:-1],valuation[i],color=colors[i],label='$\hat v_{%d}(t)$'%(i+1),
            **dual_style)
    ax.fill_between(t[:-1],input_active[i],color=colors[i],**u_active_style)
ax.set_xlabel('Time $t$ [s]')
ax.set_ylabel('Normalized valuation $\hat v_i(t)$')
ax.legend()
plt.tight_layout()
ax.autoscale(tight=True)

fig = plt.figure(5,figsize=(6.4,4.8))
plt.clf()
ax = fig.add_subplot(111)
for i in range(M):
    ax.plot(t[:-1],cost_coeff[i],color=colors[i],
            label='$\hat\\beta_2^{%d}(t)-\hat\\beta_3^{%d}(t)$'%(i+1,i+1),**dual_style)
ax.set_xlabel('Time $t$ [s]')
ax.set_ylabel('Normalized $\hat\\beta_2^i(t)-\hat\\beta_3^i(t)$')
ax.legend()
plt.tight_layout()
ax.autoscale(tight=True)
ylims = ax.get_ylim()
for i in range(M):
    ax.fill_between(t[:-1],input_active[i]*ylims[0],input_active[i]*ylims[1],
                    color=colors[i],**u_active_style)
ax.axhline(0,color='black',linestyle='--',zorder=0)

primer_style = dict(color='black')
primer_style_ic = dict(color='black',marker='x',linestyle='none')
equiv_plane_style = gs_style
yscale = 1e-3

fig = plt.figure(6,figsize=(6.4,4.8))
plt.clf()
ax = fig.add_subplot(111)
ax.axis('equal')
for i in range(M):
    draw_polar_grid(ax,theta[i],[rho1[i],rho2[i]],np.array([0,1]),
                    angles,magnitudes,
                    border_style[i],grid_style[i],labels=False)
    ax.plot([u if la.norm(u)>np.sqrt(1e-5) else np.nan for u in primal['u'][i][0]],
            [u if la.norm(u)>np.sqrt(1e-5) else np.nan for u in primal['u'][i][1]],
            label='$u_{%d}(t)$'%(i),**thrust_style[i])
ax.plot(0,0,**origin_style)
ax.plot(yscale*y[:,0],yscale*y[:,1],label='$y(t)$',**primer_style)
ax.plot(yscale*y[0,0],yscale*y[0,1],label='$y(0)$',**primer_style_ic)
ax.set_xlabel('$x$-acceleration $u_{i,1}(t)$ [m/s$^2$]')
ax.set_ylabel('$y$-acceleration $u_{i,2}(t)$ [m/s$^2$]')
plt.tight_layout()
#ax.autoscale(tight=True)
fig.canvas.draw()
ylims = ax.get_ylim()
xlims = ax.get_xlim()
s = xlims[1]/n_equiv[0]
ax.plot([0,s*n_equiv[0]],[0,s*n_equiv[1]],**equiv_plane_style)
ax.plot([0,-s*n_equiv[0]],[0,s*n_equiv[1]],**equiv_plane_style)
ax.set_xlim(xlims)
ax.set_ylim(ylims)
ax.legend()