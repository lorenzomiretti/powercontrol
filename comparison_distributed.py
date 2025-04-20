# -*- coding: utf-8 -*-
"""
Code for "Two-timescale joint power control and beamforming design with applications to cell-free massive MIMO"
Author: Lorenzo Miretti

Output: Figure 5 - comparison of distributed cell-free schemes

License: This code is licensed under the GPLv2 license. If you in any way
use this code for research that results in publications, please cite our
paper as described in the README file
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm
import powercontrol as pc

def main():
    # Parameters
    N_iter_max = 30          # Number of iterations
    N_cdf = 100              # Number of user drops
    weights = np.ones(pc.K)  # Weights of max-min problem
    toll = 1e-1              # Algorithm convergence tolerance 

    # Initialization
    np.random.seed(0)
    R_pc = np.array([])
    R_MF = np.array([])
    R_LTMMSE = np.array([])
    for n in range(N_cdf):
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
        _, p_LTMMSE, parameters_LTMMSE = pc.normalized_FP_iterations(H_hat_list,Err_cov_list,clusters,pc.LTMMSE,weights,N_iter_max,toll)
        # Evaluate performance (coherent bound)
        R_LTMMSE = np.append(R_LTMMSE,pc.compute_ergodic_rates(H_list,H_hat_list,p_LTMMSE,pc.LTMMSE,parameters_LTMMSE))

        # Long-term max-min power control with fixed beamforming design
        p_full = np.ones(pc.K)*pc.p_max
        C_list_pc, Psi_list_pc = pc.compute_parameters_LTMMSE(H_hat_list,Err_cov_list,clusters,p_full)
        parameters_pc = (C_list_pc, Psi_list_pc,p_full) 
        sqrtErr_cov_list = [[sqrtm(Err_cov_list[l][k]) for k in range(pc.K)] for l in range(pc.L)]
        interf, signal, noise = pc.compute_channel_coefficients(H_hat_list,sqrtErr_cov_list,pc.LTMMSE,parameters_pc)
        p_pc, _ = pc.max_min_power_control(interf,signal/weights,noise,pc.p_max)
        # Evaluate performance (coherent bound rates)
        R_pc = np.append(R_pc, pc.compute_ergodic_rates(H_list,H_hat_list,p_pc,pc.LTMMSE,parameters_pc))

        # Joint LSFD and power control (MF)
        _, p_MF, parameters_MF = pc.normalized_FP_iterations(H_hat_list,Err_cov_list,clusters,pc.MF,weights,N_iter_max, toll)
        # Evaluate performance (coherent bound rates)
        R_MF = np.append(R_MF,pc.compute_ergodic_rates(H_list,H_hat_list,p_MF,pc.MF,parameters_MF))

        print(n+1,'/', N_cdf)

    # Plot 
    plt.figure(figsize=(8, 5))
    fontSize = 15
    lwidth = 4
    msize = 10
    N_markers = 10
    marker_sep = round(pc.K*N_cdf/N_markers)
    y_cdf = np.arange(1,pc.K*N_cdf+1)/pc.K/N_cdf
    R_LTMMSE.sort()
    plt.plot(R_LTMMSE, y_cdf ,'-^', lw = lwidth, ms = msize, markevery=marker_sep,label='Joint beamforming')
    R_MF.sort()
    plt.plot(R_MF, y_cdf ,'-o', lw = lwidth, ms = msize, markevery=marker_sep,label='Joint LSFD (MRC)')
    R_pc.sort()
    plt.plot(R_pc, y_cdf ,'-s', lw = lwidth, ms = msize, markevery=marker_sep,label='Power control')
    plt.xlabel('Rate [b/s/Hz]', fontsize=fontSize)
    plt.ylabel('CDF', fontsize=fontSize)
    plt.xticks(fontsize = fontSize-2)
    plt.yticks(fontsize = fontSize-2)
    plt.xlim((0,4))
    plt.grid()
    plt.tight_layout()
    plt.legend(fontsize= fontSize)
    plt.show()

main()