import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import math
from math import pi as pi
from scipy import optimize

import matplotlib as mpl
from matplotlib import cm
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'

import uposham.turning_point as tp
import voter97_hamiltonian as voter97

import os
path_to_data = os.path.join(os.path.dirname(os.path.dirname(__file__)), \
                            'data/')
path_to_saveplot = os.path.join(os.path.dirname(os.path.dirname(__file__)), \
                                'tests/plots/')



#%% Setting up parameters and global variables

save_final_plot = True
show_final_plot = False
show_itrsteps_plots = False # show iteration of the UPOs in plots
N = 4          # dimension of phase space
MASS_A = 1
MASS_B = 1
d_1 = 4
d_2 = 2*pi
SADDLE_ENERGY = 1.0 #Energy of the saddle

parameters = np.array([MASS_A, MASS_B, d_1, d_2])
eqNum = 1  
eqPt = eqPt = [1, -parameters[2]/(2*pi*parameters[3])]


deltaE_vals = [0.1, 1.00]
xLeft = [0.0,0.01]
xRight = [0.05,0.11]
linecolor = ['b','r']

for i in range(len(deltaE_vals)):
    
    e = deltaE_vals[i]
    n = 12
    n_turn = 1
    deltaE = e
    
    #Trial initial Condition s.t. one initial condition is on the LHS of the UPO and the 
    #other one is on the RHS of the UPO

    x = xLeft[i]
    f2 = lambda y: voter97.get_coord_voter97(x, y, e, parameters)

    yanalytic = (-np.cos(2*pi*x)*parameters[2] - np.sqrt((np.cos(2*pi*x)*parameters[2])**2 + \
                    4*parameters[3]*pi*e))/(2*parameters[3]*pi)

    state0_2 = [x,optimize.newton(f2,yanalytic),0.0,0.0]
    
    x = xRight[i]
    f3 = lambda y: voter97.get_coord_voter97(x, y, e, parameters)

    yanalytic = (-np.cos(2*pi*x)*parameters[2] - np.sqrt((np.cos(2*pi*x)*parameters[2])**2 + \
                    4*parameters[3]*pi*e))/(2*parameters[3]*pi)

    state0_3 = [x, optimize.newton(f3,yanalytic),0.0,0.0]
    
    
    with open("x0_turningpoint_deltaE%s_coupled.dat" %(deltaE),'a+') as po_fam_file:
        [x0po_1, T_1,energyPO_1] = tp.turningPoint(state0_2, state0_3, \
                                            voter97.get_coord_voter97, \
                                            voter97.guess_coords_voter97, \
                                            voter97.ham2dof_voter97, \
                                            voter97.half_period_voter97, \
                                            voter97.variational_eqns_voter97, \
                                            voter97.pot_energy_voter97, \
                                            voter97.plot_iter_orbit_voter97, \
                                            parameters, \
                                            e, n, n_turn, show_itrsteps_plots, \
                                            po_fam_file) 



#%% Load periodic orbit data from ascii files
    
x0po = np.zeros((4,len(deltaE_vals))) #each column is a different initial condition

for i in range(len(deltaE_vals)):
    deltaE = deltaE_vals[i]

    with open("x0_turningpoint_deltaE%s_voter97.dat" %(deltaE),'a+') as po_fam_file:
        print('Loading the periodic orbit family from data file',po_fam_file.name,'\n') 
        x0podata = np.loadtxt(po_fam_file.name)
        x0po[:,i] = x0podata[-1,0:4] 


#%% Plotting the Family

TSPAN = [0,30]
plt.close('all')
axis_fs = 15
RelTol = 3.e-10
AbsTol = 1.e-10

f = lambda t,x : voter97.ham2dof_voter97(t,x,parameters) 

ax = plt.gca(projection='3d')

for i in range(len(deltaE_vals)):
    
    soln = solve_ivp(f, TSPAN, x0po[:,i], method='RK45', dense_output=True, \
                    events = lambda t,x : voter97.half_period_voter97(t,x,parameters), \
                    rtol=RelTol, atol=AbsTol)
    te = soln.t_events[0]
    tt = [0,te[2]]
    t,x,phi_t1,PHI = tp.state_transit_matrix(tt, x0po[:,i], parameters, \
                    voter97.variational_eqns_voter97)
    
    
    ax.plot(x[:,0],x[:,1],x[:,3],'-', color=linecolor[i], \
            label='$\Delta E$ = %.2f'%(deltaE_vals[i]))
    ax.scatter(x[0,0],x[0,1],x[0,3], s=20, marker='*')
    ax.plot(x[:,0], x[:,1], zs=0, zdir='z')


resX = 100
xVec = np.linspace(-4,4,resX)
yVec = np.linspace(-4,4,resX)
xMat, yMat = np.meshgrid(xVec, yVec)
cset1 = ax.contour(xMat, yMat, 
                   tp.get_pot_surf_proj(xVec, yVec, \
                        voter97.pot_energy_voter97, parameters), \
                    [0.01,0.1,1,2,4], zdir='z', offset=0, \
                    linewidths = 1.0, cmap=cm.viridis, \
                    alpha = 0.8)

ax.scatter(eqPt[0], eqPt[1], s = 200, c = 'r', marker = 'X')
ax.set_xlabel('$x$', fontsize=axis_fs)
ax.set_ylabel('$y$', fontsize=axis_fs)
ax.set_zlabel('$p_y$', fontsize=axis_fs)

legend = ax.legend(loc='upper left')
ax.set_xlim(-4, 4)
ax.set_ylim(-4, 4)
ax.set_zlim(-4, 4)

plt.grid()


if show_final_plot:
    plt.show()

if save_final_plot:  
    plt.savefig('./tests/plots/tp_voter97_upos.pdf', format='pdf', \
                bbox_inches='tight')
