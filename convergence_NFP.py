# -*- coding: utf-8 -*-
"""
Code for "Two-timescale joint power control and beamforming design with applications to cell-free massive MIMO"
Author: Lorenzo Miretti

Output: Figure 2b - convergence of normalized fixed-point iterations

License: This code is licensed under the GPLv2 license. If you in any way
use this code for research that results in publications, please cite our
paper as described in the README file
"""

import numpy as np
import matplotlib.pyplot as plt
import powercontrol as pc

def main():
    # Parameters
    N_iter_max = 20                         # Number of iterations
    weights = np.ones(pc.K)                 # Weights of max-min problem
    # Initialization
    np.random.seed(0)
    # Generate AP and random UE positions
    pos = pc.generate_setup(plot = False)
    # Compute path loss
    Gamma = pc.compute_ch_statistics(pos)
    # Draw a list of N_sim channel realizations
    H_list = pc.draw_channel_realizations(Gamma)

    # Cell-free schemes
    # Compute user-centric clusters (cluster size Q <= L)        
    clusters = pc.compute_clusters(Gamma,Q=4)
    # Compute channel estimates
    H_hat_list = pc.draw_CSI_realizations(H_list,clusters)
    # Compute error covariances 
    Err_cov_list = pc.compute_error_covariances(Gamma,clusters)
    # Solve max-min rate problem
    p_MMSE, _, _ = pc.normalized_FP_iterations(H_hat_list,Err_cov_list,clusters,pc.MMSE,weights,N_iter_max,toll=0)
    p_LTMMSE, _, _ = pc.normalized_FP_iterations(H_hat_list,Err_cov_list,clusters,pc.LTMMSE,weights,N_iter_max,toll=0)

    # Small cells (need to recompute the channel estimates)
    # Define cells 
    clusters = pc.compute_clusters(Gamma,Q=1)
    # Draw a list of N_sim local MMSE channel estimates, and get error covariances
    H_hat_list = pc.draw_CSI_realizations(H_list,clusters)
    # Compute error covariances 
    Err_cov_list = pc.compute_error_covariances(Gamma,clusters)
    # Solve max-min rate problem
    p_cell, _, _ = pc.normalized_FP_iterations(H_hat_list,Err_cov_list,clusters,pc.MMSE,weights,N_iter_max,toll=0)

    # Plot
    plt.figure(figsize=(8, 5))
    fontSize = 15
    lwidth = 4
    msize = 10
    iters = np.arange(len(p_MMSE)-1)
    dist_MMSE = [np.linalg.norm(p_MMSE[n]-p_MMSE[-1]) for n in iters]
    dist_LTMMSE = [np.linalg.norm(p_LTMMSE[n]-p_LTMMSE[-1]) for n in iters]
    dist_cell = [np.linalg.norm(p_cell[n]-p_cell[-1]) for n in iters]
    plt.semilogy(iters,dist_MMSE,'o--',lw=lwidth,ms = msize,label='Centralized')
    plt.semilogy(iters,dist_LTMMSE,'^--',lw=lwidth,ms = msize,label='Distributed')
    plt.semilogy(iters,dist_cell,'s--',lw=lwidth, label='Small-cells')
    plt.xlabel('Iteration number', fontsize = fontSize)
    plt.ylabel('Distance from solution', fontsize = fontSize)
    plt.xticks(np.arange(0,len(p_MMSE),2),fontsize = fontSize-2)
    plt.yticks(fontsize = fontSize-2)
    plt.grid()
    plt.ylim((10**-4,10**3))
    plt.xlim((0,16))
    plt.tight_layout()
    plt.legend(fontsize= fontSize)
    plt.show()

main()