# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 10:02:48 2019

@author: wl16298
"""

"""Plot peridoic orbits with 5 different energies using the new method"""

# For deleonberne problem

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint,quad,trapz,solve_ivp
import math
from IPython.display import Image # for Notebook
from IPython.core.display import HTML
from mpl_toolkits.mplot3d import Axes3D
from numpy import linalg as LA
import scipy.linalg as linalg
from scipy.optimize import fsolve
import time
from functools import partial
import deleonberne_newmethod ### import module xxx where xxx is the name of the python file xxx.py 
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
from matplotlib import cm
from pylab import rcParams
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'
#%matplotlib
from scipy import optimize
#%% Setting up parameters and global variables
N = 4;          # dimension of phase space
MASS_A = 8.0; MASS_B = 8.0; # De Leon, Marston (1989)
EPSILON_S = 1.0;
D_X = 10.0;
ALPHA = 1.00;
LAMBDA = 1.5;
parameters = np.array([MASS_A, MASS_B, EPSILON_S, D_X, LAMBDA, ALPHA]);
eqNum = 1;  
eqPt = deleonberne_newmethod.get_eq_pts_deleonberne(eqNum, parameters)
#%%
# e is the total energy, n is the number of intervals we want to divide, n_turn is the nth turning point we want to choose.
e=1.10
n=4
n_turn = 1
deltaE = e-parameters[2]  # In this case deltaE is 0.10
"""Trial initial Condition s.t. one initial condition is on the LHS of the UPO and the other one is on the RHS of the UPO"""
f1 = partial(deleonberne_newmethod.get_x,y=0.06,V=e,par=parameters)
x0_2 = optimize.newton(f1,-0.15)
state0_2 = [x0_2,0.06,0.0,0.0]
f2 = partial(deleonberne_newmethod.get_x,y=-0.05,V=e,par=parameters)
x0_3 = optimize.newton(f2,-0.15)
state0_3 = [x0_3, -0.05,0.0,0.0]

po_fam_file = open("1111x0_newmethod_deltaE%s_deleonberne.txt" %(0.10),'a+')
[x0po_1, T_1,energyPO_1] = deleonberne_newmethod.newmethod_deleonberne(state0_3,state0_2 ,parameters,e,n,n_turn,po_fam_file) ; 

po_fam_file.close()
#%%
# e is the total energy, n is the number of intervals we want to divide, n_turn is the nth turning point we want to choose.
e=2.0
n=4
n_turn = 1
deltaE = e-parameters[2]
"""Trial initial Condition s.t. one initial condition is on the LHS of the UPO and the other one is on the RHS of the UPO"""
f1 = partial(deleonberne_newmethod.get_x,y=0.06,V=e,par=parameters)
x0_2 = optimize.newton(f1,-0.15)
state0_2 = [x0_2,0.06,0.0,0.0]
f2 = partial(deleonberne_newmethod.get_x,y=-0.05,V=e,par=parameters)
x0_3 = optimize.newton(f2,-0.15)
state0_3 = [x0_3, -0.05,0.0,0.0]

po_fam_file = open("1111x0_newmethod_deltaE%s_deleonberne.txt" %(deltaE),'a+')
[x0po_2, T_2,energyPO_2] = deleonberne_newmethod.newmethod_deleonberne(state0_3,state0_2 ,parameters,e,n,n_turn,po_fam_file) ; 

po_fam_file.close()
#%%
# e is the total energy, n is the number of intervals we want to divide, n_turn is the nth turning point we want to choose.
e=3.0
n=4
n_turn = 1
deltaE = e-parameters[2]
"""Trial initial Condition s.t. one initial condition is on the LHS of the UPO and the other one is on the RHS of the UPO"""
f1 = partial(deleonberne_newmethod.get_x,y=0.06,V=e,par=parameters)
x0_2 = optimize.newton(f1,-0.15)
state0_2 = [x0_2,0.06,0.0,0.0]
f2 = partial(deleonberne_newmethod.get_x,y=-0.05,V=e,par=parameters)
x0_3 = optimize.newton(f2,-0.15)
state0_3 = [x0_3, -0.05,0.0,0.0]

po_fam_file = open("1111x0_newmethod_deltaE%s_deleonberne.txt" %(deltaE),'a+')
[x0po_3, T_3,energyPO_3] = deleonberne_newmethod.newmethod_deleonberne(state0_3,state0_2 ,parameters,e,n,n_turn,po_fam_file) ; 

po_fam_file.close()
#%%
# e is the total energy, n is the number of intervals we want to divide, n_turn is the nth turning point we want to choose.
e=5.0
n=4
n_turn = 1
deltaE = e-parameters[2]
"""Trial initial Condition s.t. one initial condition is on the LHS of the UPO and the other one is on the RHS of the UPO"""
f1 = partial(deleonberne_newmethod.get_x,y=0.06,V=e,par=parameters)
x0_2 = optimize.newton(f1,-0.15)
state0_2 = [x0_2,0.06,0.0,0.0]
f2 = partial(deleonberne_newmethod.get_x,y=-0.05,V=e,par=parameters)
x0_3 = optimize.newton(f2,-0.15)
state0_3 = [x0_3, -0.05,0.0,0.0]

po_fam_file = open("1111x0_newmethod_deltaE%s_deleonberne.txt" %(deltaE),'a+')
[x0po_4, T_4,energyPO_4] = deleonberne_newmethod.newmethod_deleonberne(state0_3,state0_2 ,parameters,e,n,n_turn,po_fam_file) ; 

po_fam_file.close()

#%% Load Data
deltaE = 0.10
po_fam_file = open("1111x0_newmethod_deltaE%s_deleonberne.txt" %(deltaE),'a+');
print('Loading the periodic orbit family from data file',po_fam_file.name,'\n'); 
x0podata = np.loadtxt(po_fam_file.name)
po_fam_file.close()
x0po_1 = x0podata

deltaE = 1.0
po_fam_file = open("1111x0_newmethod_deltaE%s_deleonberne.txt" %(deltaE),'a+');
print('Loading the periodic orbit family from data file',po_fam_file.name,'\n'); 
x0podata = np.loadtxt(po_fam_file.name)
po_fam_file.close()
x0po_2 = x0podata

deltaE = 2.0
po_fam_file = open("1111x0_newmethod_deltaE%s_deleonberne.txt" %(deltaE),'a+');
print('Loading the periodic orbit family from data file',po_fam_file.name,'\n'); 
x0podata = np.loadtxt(po_fam_file.name)
po_fam_file.close()
x0po_3 = x0podata

deltaE = 4.0
po_fam_file = open("1111x0_newmethod_deltaE%s_deleonberne.txt" %(deltaE),'a+');
print('Loading the periodic orbit family from data file',po_fam_file.name,'\n'); 
x0podata = np.loadtxt(po_fam_file.name)
po_fam_file.close()
x0po_4 = x0podata


#%% Plotting the Family
TSPAN = [0,30]
plt.close('all')
axis_fs = 15
RelTol = 3.e-10
AbsTol = 1.e-10
f = partial(deleonberne_newmethod.deleonberne2dof, par=parameters) 
soln = solve_ivp(f, TSPAN, x0po_1[-1,0:4],method='RK45',dense_output=True, events = deleonberne_newmethod.half_period,rtol=RelTol, atol=AbsTol)
te = soln.t_events[0]
tt = [0,te[2]]
t,x,phi_t1,PHI = deleonberne_newmethod.stateTransitMat_deleonberne(tt,x0po_1[-1,0:4],parameters)
ax = plt.gca(projection='3d')
ax.plot(x[:,0],x[:,1],x[:,2],'-',label='$\Delta E$ = 0.1')
ax.scatter(x[0,0],x[0,1],x[0,2],s=20,marker='*');
ax.plot(x[:,0], x[:,1], zs=0, zdir='z')


f = partial(deleonberne_newmethod.deleonberne2dof, par=parameters) 
soln = solve_ivp(f, TSPAN, x0po_2[-1,0:4],method='RK45',dense_output=True, events = deleonberne_newmethod.half_period,rtol=RelTol, atol=AbsTol)
te = soln.t_events[0]
tt = [0,te[2]]
t,x,phi_t1,PHI = deleonberne_newmethod.stateTransitMat_deleonberne(tt,x0po_2[-1,0:4],parameters)
ax = plt.gca(projection='3d')
ax.plot(x[:,0],x[:,1],x[:,2],'-',label='$\Delta E$ = 1.0')
ax.scatter(x[0,0],x[0,1],x[0,2],s=20,marker='*');
ax.plot(x[:,0], x[:,1], zs=0, zdir='z')

f = partial(deleonberne_newmethod.deleonberne2dof, par=parameters) 
soln = solve_ivp(f, TSPAN, x0po_3[-1,0:4],method='RK45',dense_output=True, events = deleonberne_newmethod.half_period,rtol=RelTol, atol=AbsTol)
te = soln.t_events[0]
tt = [0,te[2]]
t,x,phi_t1,PHI = deleonberne_newmethod.stateTransitMat_deleonberne(tt,x0po_3[-1,0:4],parameters)
ax = plt.gca(projection='3d')
ax.plot(x[:,0],x[:,1],x[:,2],'-',label='$\Delta E$ = 2.0')
ax.scatter(x[0,0],x[0,1],x[0,2],s=20,marker='*');
ax.plot(x[:,0], x[:,1], zs=0, zdir='z')


f = partial(deleonberne_newmethod.deleonberne2dof, par=parameters) 
soln = solve_ivp(f, TSPAN, x0po_4[-1,0:4],method='RK45',dense_output=True, events = deleonberne_newmethod.half_period,rtol=RelTol, atol=AbsTol)
te = soln.t_events[0]
tt = [0,te[2]]
t,x,phi_t1,PHI = deleonberne_newmethod.stateTransitMat_deleonberne(tt,x0po_4[-1,0:4],parameters)
ax = plt.gca(projection='3d')
ax.plot(x[:,0],x[:,1],x[:,2],'-',label='$\Delta E$ = 4.0')
ax.scatter(x[0,0],x[0,1],x[0,2],s=20,marker='*');
ax.plot(x[:,0], x[:,1], zs=0, zdir='z')


ax = plt.gca(projection='3d')
resX = 100
xVec = np.linspace(-1,1,resX)
yVec = np.linspace(-2,2,resX)
xMat, yMat = np.meshgrid(xVec, yVec)
cset1 = ax.contour(xMat, yMat, deleonberne_newmethod.get_pot_surf_proj(xVec, yVec,parameters), [1.1,2,3,5],zdir='z', offset=0,
                       linewidths = 1.0, cmap=cm.viridis, alpha = 0.8)

ax.scatter(eqPt[0], eqPt[1], s = 200, c = 'r', marker = 'X')
ax.set_xlabel('$x$', fontsize=axis_fs)
ax.set_ylabel('$y$', fontsize=axis_fs)
ax.set_zlabel('$v_x$', fontsize=axis_fs)
#ax.set_title('$\Delta E$ = %1.e,%1.e,%1.e,%1.e,%1.e' %(energyPO_1[-1]-parameters[2],energyPO_2[-1]-parameters[2],energyPO_3[-1]-parameters[2],energyPO_4[-1]-parameters[2],energyPO_5[-1]-parameters[2]) ,fontsize=axis_fs)
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_zlim(-4, 4)
legend = ax.legend(loc='best')
plt.savefig('newmethod_POfam_deleonberne',format='pdf',bbox_inches='tight')
plt.grid()
plt.show()