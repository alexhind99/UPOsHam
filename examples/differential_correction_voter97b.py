# -*- coding: utf-8 -*-
# """

# @author: Alexander Hind

# Script to compute unstable periodic orbits at specified energies for the voter97 Hamiltonian using differential correction
# """

import numpy as np

from scipy.integrate import solve_ivp

import time

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
from math import pi as pi
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'

import uposham.differential_correction as diffcorr
import voter97_hamiltonian as voter97

import os
path_to_data = os.path.join(os.path.dirname(os.path.dirname(__file__)), \
                            'data/')
path_to_saveplot = os.path.join(os.path.dirname(os.path.dirname(__file__)), \
                                'tests/plots/')

N = 4          # dimension of phase space
MASS_A = 1.00
MASS_B = 1.00
SADDLE_ENERGY = 1.0 #Energy of the saddle

#Optimising Transition states parameters
d_1 = 0   #Coupling term
d_2 = 1

parameters = np.array([MASS_A, MASS_B, \
                       d_1, d_2, SADDLE_ENERGY])
eqNum = 1
     

def upo(deltaE_vals, linecolor, \
        save_final_plot = True, show_final_plot = True):
    # Setting up parameters and global variables

    eqPt = [1, -parameters[2]/(2*pi*parameters[3])]
    #energy of the saddle eq pt
    eSaddle = 1


    
    nFam = 100 # use nFam = 10 for low energy

    # first two amplitudes for continuation procedure to get p.o. family
    Ax1  = 2.e-5 # initial amplitude (1 of 2) values to use: 2.e-3
    Ax2  = 2*Ax1 # initial amplitude (2 of 2)

    t = time.time()

    # get the initial conditions and periods for a family of periodic orbits
    with open("x0_diffcorr_fam_eqPt%s_voter97.dat" %eqNum,'a+') as po_fam_file:
        [po_x0Fam,po_tpFam] = diffcorr.get_po_fam(
            eqNum, Ax1, Ax2, nFam, po_fam_file, voter97.init_guess_eqpt_voter97, \
            voter97.grad_pot_voter97, voter97.jacobian_voter97, \
            voter97.guess_lin_voter97, voter97.diffcorr_setup_voter97, \
            voter97.conv_coord_voter97, voter97.diffcorr_acc_corr_voter97, \
            voter97.ham2dof_voter97, voter97.half_period_voter97, \
            voter97.pot_energy_voter97, voter97.variational_eqns_voter97, \
            voter97.plot_iter_orbit_voter97, parameters)

        poFamRuntime = time.time()-t
        x0podata = np.concatenate((po_x0Fam, po_tpFam),axis=1)


    

    for i in range(len(deltaE_vals)):
        deltaE = deltaE_vals[i]
        
        with open("x0_diffcorr_fam_eqPt%s_voter97.dat" %eqNum ,'a+') as po_fam_file:
            eTarget = eSaddle + deltaE 
            print('Loading the periodic orbit family from data file',po_fam_file.name,'\n') 
            x0podata = np.loadtxt(po_fam_file.name)
        
        
        #%
        with open("x0po_T_energyPO_eqPt%s_brac%s_voter97.dat" %(eqNum,deltaE),'a+') as po_brac_file:
            t = time.time()
        
            x0poTarget,TTarget = diffcorr.po_bracket_energy(
                eTarget, x0podata, po_brac_file, \
                voter97.diffcorr_setup_voter97, voter97.conv_coord_voter97, \
                voter97.diffcorr_acc_corr_voter97, voter97.ham2dof_voter97, \
                voter97.half_period_voter97, voter97.pot_energy_voter97, \
                voter97.variational_eqns_voter97, voter97.plot_iter_orbit_voter97, \
                parameters)

            poTarE_runtime = time.time()-t
            with open(
                "model_parameters_eqPt%s_DelE%s_voter97.dat" %(eqNum,deltaE),'a+') as model_parameters_file:
                np.savetxt(model_parameters_file.name, parameters,fmt='%1.16e')
        
        
        # target specific periodic orbit
        # Target PO of specific energy with high precision does not work for the
        # model 
        
        with open("x0_diffcorr_deltaE%s_voter97.dat" %(deltaE),'a+')as po_target_file:

            [x0po, T,energyPO] = diffcorr.po_target_energy(
                x0poTarget,eTarget, po_target_file, \
                voter97.diffcorr_setup_voter97, voter97.conv_coord_voter97, \
                voter97.diffcorr_acc_corr_voter97, \
                voter97.ham2dof_voter97, voter97.half_period_voter97, \
                voter97.pot_energy_voter97, voter97.variational_eqns_voter97, \
                voter97.plot_iter_orbit_voter97, parameters)


    #Load periodic orbit data from ascii files
        
    x0po = np.zeros((4,len(deltaE_vals)))

    for i in range(len(deltaE_vals)):
        deltaE = deltaE_vals[i]

        with open("x0_diffcorr_deltaE%s_voter97.dat" %(deltaE),'a+') as po_fam_file:
            print('Loading the periodic orbit family from data file',po_fam_file.name,'\n') 
            x0podata = np.loadtxt(po_fam_file.name)
        x0po[:,i] = x0podata[0:4]


    # Plotting the unstable periodic orbits at the specified energies

    TSPAN = [0,30] # arbitrary range, just to finish the integration
    plt.close('all')
    axis_fs = 15
    RelTol = 3.e-10
    AbsTol = 1.e-10

    for i in range(len(deltaE_vals)):
        deltaE = deltaE_vals[i]
        f= lambda t,x: voter97.ham2dof_voter97(t,x,parameters)
        soln = solve_ivp(f, TSPAN, x0po[:,i],method='RK45',dense_output=True, \
                        events = lambda t,x : voter97.half_period_voter97(
                            t,x,parameters), \
                        rtol=RelTol, atol=AbsTol)
        
        te = soln.t_events[0]
        tt = [0,te[1]]
        t,x,phi_t1,PHI = diffcorr.state_transit_matrix(tt, x0po[:,i], parameters, \
                                                voter97.variational_eqns_voter97)
        ax = plt.gca(projection='3d')
        ax.plot(x[:,0],x[:,1],x[:,3],'-',color = linecolor[i], \
                label = '$\Delta E$ = %.2f'%(deltaE))
        ax.plot(x[:,0],x[:,1],-x[:,3],'-',color = linecolor[i])
        ax.scatter(x[0,0],x[0,1],x[0,3],s=10,marker='*')
        ax.scatter(x[0,0],x[0,1],-x[0,3],s=10,marker='o')
        ax.plot(x[:,0], x[:,1], zs=0, zdir='z')


    ax = plt.gca(projection='3d')
    resX = 100
    xVec = np.linspace(0,3,resX)
    yVec = np.linspace(-2,2,resX)
    xMat, yMat = np.meshgrid(xVec, yVec)
    cset1 = ax.contour(
            xMat, yMat, diffcorr.get_pot_surf_proj(
                xVec, yVec, voter97.pot_energy_voter97, \
                parameters), \
            [0.01,0.1,1,2,4], zdir='z', offset=0, linewidths = 1.0, \
            cmap=cm.viridis, alpha = 0.8)

    ax.scatter(eqPt[0], eqPt[1], s = 50, c = 'r', marker = 'X')
    ax.set_xlabel('$x$', fontsize=axis_fs)
    ax.set_ylabel('$y$', fontsize=axis_fs)
    ax.set_zlabel('$p_y$', fontsize=axis_fs)


    legend = ax.legend(loc='upper left')
    ax.set_xlim(0, 3)
    ax.set_ylim(-2, 2)
    ax.set_zlim(-2, 2)

    plt.grid()

    if show_final_plot:
        plt.show()

    if save_final_plot:  
        plt.savefig(path_to_saveplot + 'diff_corr_coupled_voter97_upos.pdf', \
                    format='pdf', bbox_inches='tight')



if __name__ == '__main__':

    deltaE_vals = [0.1]
    linecolor = ['r']

    upo(deltaE_vals, linecolor)