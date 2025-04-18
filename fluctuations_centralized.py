# -*- coding: utf-8 -*-
"""
Code for "Two-timescale joint power control and beamforming design with applications to cell-free massive MIMO"
Author: Lorenzo Miretti

Output: Figure 4b - instantaneous rate fluctuations of centralized cell-free schemes 

License: This code is licensed under the GPLv2 license. If you in any way
use this code for research that results in publications, please cite our
paper as described in the README file
"""

import numpy as np
import matplotlib.pyplot as plt
import powercontrol as pc

def main():
    # Parameters
    N_iter_max = 30         # Number of iterations
    weights = np.ones(pc.K)
    toll = 1e-2

    # Initialization
    np.random.seed(0)
    R_MMSE = np.array([])
    R_MMSE_short = np.array([])

    # Generate AP and random UE positions
    pos = pc.generate_setup(plot = False)
    # Compute path loss
    Gamma = pc.compute_ch_statistics(pos)
    # Draw a list of N_sim channel realizations
    H_list = pc.draw_channel_realizations(Gamma)
    # Compute user-centric clusters (cluster size Q <= L)        
    clusters = pc.compute_clusters(Gamma,Q=4)
    # Compute channel estimates
    H_hat_list = pc.draw_CSI_realizations(H_list,clusters)
    # Compute error covariances 
    Err_cov_list = pc.compute_error_covariances(Gamma,clusters)

   # Long-term max-min joint problem
    _, p_MMSE, parameters_MMSE = pc.normalized_FP_iterations(H_hat_list,Err_cov_list,clusters,pc.MMSE,weights,N_iter_max,toll)
    # Evaluate instataneous rates with imperfect CSI 
    R_MMSE = np.zeros((pc.K,pc.N_sim))
    for m in range(pc.N_sim):
        # UatF bound with the expectations calculated using a single sample
        R_MMSE[:,m] += np.array(pc.compute_UatF_rates([H_hat_list[m]],Err_cov_list,p_MMSE,pc.MMSE,parameters_MMSE))

    # Short-term max-min rate problem 
    R_MMSE_short = np.zeros((pc.K,pc.N_sim))
    for m in range(pc.N_sim):
        _, p_MMSE, parameters_MMSE = pc.normalized_FP_iterations([H_hat_list[m]],Err_cov_list,clusters,pc.MMSE,weights,N_iter_max,toll)
        # UatF bound with the expectations calculated using a single sample
        R_MMSE_short[:,m] = np.array(pc.compute_UatF_rates([H_hat_list[m]],Err_cov_list,p_MMSE,pc.MMSE,parameters_MMSE))

    # Plot
    plt.figure(figsize=(8, 5))
    fontSize = 12
    lwidth = 2
    msize = 8
    N_pts = 25
    line1, = plt.plot(np.arange(1,N_pts+1),R_MMSE[0,:N_pts],'-o', lw = lwidth, ms = msize,label='Long-term joint (user #1)')
    color1 = line1.get_color()
    plt.plot(np.arange(1,N_pts+1), R_MMSE[1,:N_pts],'--x', lw = lwidth, color = color1, ms = msize,label='Long-term joint (user #2)')
    plt.plot(np.arange(1,N_pts+1), R_MMSE_short[0,:N_pts],'-^', lw = lwidth, ms = msize,label='Short-term joint (all users)')
    plt.ylabel('Instantaneous rate [b/s/Hz]', fontsize=fontSize)
    plt.xlabel('Channel realization', fontsize=fontSize)
    plt.grid()
    plt.tight_layout()
    plt.legend(fontsize= fontSize)
    plt.show()

main()