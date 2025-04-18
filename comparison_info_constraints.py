# -*- coding: utf-8 -*-
"""
Code for "Two-timescale joint power control and beamforming design with applications to cell-free massive MIMO"
Author: Lorenzo Miretti

Output: Figure 3 - comparison of cell-free and small cells networks

License: This code is licensed under the GPLv2 license. If you in any way
use this code for research that results in publications, please cite our
paper as described in the README file
"""

import numpy as np
import matplotlib.pyplot as plt
import powercontrol as pc

def main():
    # Parameters
    N_iter_max = 30          # Number of iterations
    N_cdf = 3              # Number of user drops
    weights = np.ones(pc.K)  # Weights of max-min problem
    toll = 1e-1              # Algorithm convergence tolerance 

    # Initialization
    np.random.seed(0)
    R_cell = np.array([])
    R_MMSE = np.array([])
    R_LTMMSE = np.array([])
    R_cell_coh = np.array([])
    R_MMSE_coh = np.array([])
    R_LTMMSE_coh = np.array([])
    for n in range(N_cdf):
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
        _, p_MMSE, parameters_MMSE = pc.normalized_FP_iterations(H_hat_list,Err_cov_list,clusters,pc.MMSE,weights,N_iter_max,toll)
        _, p_LTMMSE, parameters_LTMMSE = pc.normalized_FP_iterations(H_hat_list,Err_cov_list,clusters,pc.LTMMSE,weights,N_iter_max,toll)
        # Evaluate performance (UatF bound)
        R_MMSE= np.append(R_MMSE,pc.compute_UatF_rates(H_hat_list,Err_cov_list,p_MMSE,pc.MMSE,parameters_MMSE))
        R_LTMMSE = np.append(R_LTMMSE,pc.compute_UatF_rates(H_hat_list,Err_cov_list,p_LTMMSE,pc.LTMMSE,parameters_LTMMSE))
        # Evaluate performance (Coherent bound)
        R_MMSE_coh= np.append(R_MMSE_coh,pc.compute_ergodic_rates(H_list,H_hat_list,p_MMSE,pc.MMSE,parameters_MMSE))
        R_LTMMSE_coh = np.append(R_LTMMSE_coh,pc.compute_ergodic_rates(H_list,H_hat_list,p_LTMMSE,pc.LTMMSE,parameters_LTMMSE))

        # Small cells (need to recompute the channel estimates)
        # Define cells 
        clusters = pc.compute_clusters(Gamma,Q=1)
        # Draw a list of N_sim local MMSE channel estimates, and get error covariances
        H_hat_list = pc.draw_CSI_realizations(H_list,clusters)
        # Compute error covariances 
        Err_cov_list = pc.compute_error_covariances(Gamma,clusters)
        # Solve max-min rate problem
        _, p_cell, parameters_cell = pc.normalized_FP_iterations(H_hat_list,Err_cov_list,clusters,pc.MMSE,weights,N_iter_max,toll)
        # Evaluate performance (UatF bound)
        R_cell = np.append(R_cell,pc.compute_UatF_rates(H_hat_list,Err_cov_list,p_cell,pc.MMSE,parameters_cell))
        # Evaluate performance (Coherent bound)
        R_cell_coh = np.append(R_cell_coh,pc.compute_ergodic_rates(H_list,H_hat_list,p_cell,pc.MMSE,parameters_cell))
        
        print(n+1,'/', N_cdf)

    # Plot UatF rates
    plt.figure(figsize=(8, 5))
    fontSize = 15
    lwidth = 4
    msize = 10
    N_markers = 10
    marker_sep = round(pc.K*N_cdf/N_markers)
    y_cdf = np.arange(1,pc.K*N_cdf+1)/pc.K/N_cdf
    R_MMSE.sort()
    plt.plot(R_MMSE, y_cdf ,'-o', lw = lwidth, ms = msize, markevery=marker_sep,label='Centralized')
    R_LTMMSE.sort()
    plt.plot(R_LTMMSE, y_cdf ,'-^', lw = lwidth, ms = msize, markevery=marker_sep,label='Distributed')
    R_cell.sort()
    plt.plot(R_cell, y_cdf ,'-s', lw = lwidth, ms = msize, markevery=marker_sep,label='Small cells')
    plt.xlabel('Rate [b/s/Hz]', fontsize=fontSize)
    plt.ylabel('CDF', fontsize=fontSize)
    plt.xticks(fontsize = fontSize-2)
    plt.yticks(fontsize = fontSize-2)
    plt.xlim((0,5))
    plt.grid()
    plt.tight_layout()
    plt.legend(fontsize= fontSize)

    # Plot coherent ergodic rates
    plt.figure(figsize=(8, 5))
    fontSize = 15
    lwidth = 4
    msize = 10
    N_markers = 10
    marker_sep = round(pc.K*N_cdf/N_markers)
    y_cdf = np.arange(1,pc.K*N_cdf+1)/pc.K/N_cdf
    R_MMSE_coh.sort()
    plt.plot(R_MMSE_coh, y_cdf ,'-o', lw = lwidth, ms = msize, markevery=marker_sep,label='Centralized')
    R_LTMMSE_coh.sort()
    plt.plot(R_LTMMSE_coh, y_cdf ,'-^', lw = lwidth, ms = msize, markevery=marker_sep,label='Distributed')
    R_cell_coh.sort()
    plt.plot(R_cell_coh, y_cdf ,'-s', lw = lwidth, ms = msize, markevery=marker_sep,label='Small cells')
    plt.xlabel('Rate [b/s/Hz]', fontsize=fontSize)
    plt.ylabel('CDF', fontsize=fontSize)
    plt.xticks(fontsize = fontSize-2)
    plt.yticks(fontsize = fontSize-2)
    plt.xlim((0,5))
    plt.grid()
    plt.tight_layout()
    plt.legend(fontsize= fontSize)

    plt.show()

main()