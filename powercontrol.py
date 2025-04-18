# -*- coding: utf-8 -*-
"""
Code for "Two-timescale joint power control and beamforming design with applications to cell-free massive MIMO"
Author: Lorenzo Miretti

License: This code is licensed under the GPLv2 license. If you in any way
use this code for research that results in publications, please cite ourpython -m venv .venv
paper as described in the README file
"""

import numpy as np
from numpy import sqrt, log2, log10, diag, eye
from numpy.linalg import norm, inv, pinv
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt

N_sim = 100             # Monte Carlo samples for estimating statistical quantities
L = 4**2                # number of APs (must be a square number)
N = 8                   # number of AP antennas 
K = 64                  # number of UEs

squareLength  = 500     # lenght of squared service area [m]
p_max = 100             # UE power budget [mW]

def test():
    # Initialization
    np.random.seed(0)
    # Generate AP and random UE positions
    pos = generate_setup(plot = False)
    # Compute path loss
    Gamma = compute_ch_statistics(pos)
    # Draw a list of N_sim channel realizations
    H_list = draw_channel_realizations(Gamma)
    # Define cells (assign user to access points)
    _, cells = compute_clusters(Gamma,Q=1)
    # Compute user-centric clusters (cluster size Q <= L)        
    clusters = compute_clusters(Gamma,Q=4)
    # Compute channel estimates
    H_hat_list = draw_CSI_realizations(H_list,clusters)
    # Compute error covariances 
    Err_cov_list = compute_error_covariances(Gamma,clusters)
    # Normalized fixed-point iterations
    weights = np.ones(K)
    N_iter_max = 20
    toll = 0.1
    normalized_FP_iterations(H_hat_list,Err_cov_list,clusters,MMSE,weights,N_iter_max, toll)
    normalized_FP_iterations(H_hat_list,Err_cov_list,clusters,LTMMSE,weights,N_iter_max, toll)

def FP_iterations(H_hat_list,Err_cov_list,clusters,beamforming,weights,N_iter_max,toll):
    # Normalized fixed point iterations 
    # Initial power vector (no power control, full power)
    p_list = [np.ones(K)*p_max]
    solved = False
    n = 0
    # preinitialize some useful auxiliary variables
    sqrtErr_cov_list = [[sqrtm(Err_cov_list[l][k]) for k in range(K)] for l in range(L)]
    while solved == False:
        p = p_list[n]
        # Beamforming optimization
        if beamforming == LTMMSE:
            C_list, Psi_list = compute_parameters_LTMMSE(H_hat_list,Err_cov_list,clusters,p)
            parameters = (C_list,Psi_list,p)
        elif beamforming == MMSE:
            Psi_list = compute_parameters_MMSE(Err_cov_list,p)
            parameters = (clusters[0],Psi_list,p) 
        interf, signal, noise = compute_channel_coefficients(H_hat_list,sqrtErr_cov_list,beamforming,parameters)
        SINR = compute_SINR(interf,signal,noise,p)    
        # Power control
        p_list.append(np.array([weights[k]*p[k]/SINR[k] for k in range(K)]))
        # Stopping conditionpy
        n += 1
        progress = np.linalg.norm(p_list[n]-p_list[n-1])
        if progress < toll:
            solved = True
            print('Solved. Progress: ', progress)
        elif n == N_iter_max:
            solved = True
            print('Warning: maximum number of iterations reached. Progress: ', progress)
    return p_list, p, parameters

def normalized_FP_iterations(H_hat_list,Err_cov_list,clusters,beamforming,weights,N_iter_max,toll):
    # Normalized fixed point iterations 
    # Initial power vector (no power control, full power)
    p_list = [np.ones(K)*p_max]
    solved = False
    n = 0
    # preinitialize some useful auxiliary variables
    sqrtErr_cov_list = [[sqrtm(Err_cov_list[l][k]) for k in range(K)] for l in range(L)]
    while solved == False:
        p = p_list[n]
        # Beamforming optimization
        if beamforming == LTMMSE:
            C_list, Psi_list = compute_parameters_LTMMSE(H_hat_list,Err_cov_list,clusters,p)
            parameters = (C_list,Psi_list,p)
        elif beamforming == MMSE:
            Psi_list = compute_parameters_MMSE(Err_cov_list,p)
            parameters = (clusters[0],Psi_list,p) 
        interf, signal, noise = compute_channel_coefficients(H_hat_list,sqrtErr_cov_list,beamforming,parameters)
        SINR = compute_SINR(interf,signal,noise,p)    
        # Power control
        t = np.array([weights[k]*p[k]/SINR[k] for k in range(K)])
        p_list.append(p_max/max(t)*t)
        # Stopping conditionpy
        n += 1
        progress = np.linalg.norm(p_list[n]-p_list[n-1])
        if progress < toll:
            solved = True
            print('Solved. Progress: ', progress)
        elif n == N_iter_max:
            solved = True
            print('Warning: maximum number of iterations reached. Progress: ', progress)
    return p_list, p, parameters

def max_min_power_control(interf,signal,noise,p_max):
    A = np.eye(K)
    b = signal
    C = interf
    sig = noise
    M = inv(diag(b)) @ C.T
    u = inv(diag(b)) @ sig.reshape((K,1))
    rad_opt = 0
    for k in range(K): 
        a_k = A[k,:].reshape((K,1))
        M_k = M + (u @ a_k.T) / p_max
        rad = np.max(np.real(np.linalg.eigvals(M_k)))
        if rad > rad_opt:
            rad_opt = rad
        t_opt = 1/rad_opt
    p_opt = t_opt*inv(np.eye(K)-t_opt*M)@u
    return p_opt.reshape(K), log2(1+t_opt)

def compute_SINR(interf,signal,noise,p):
    SINR = [p[k]*signal[k]/((interf[:,k]) @ p + noise[k]) for k in range(K)]
    return SINR

def compute_UatF_rates(H_hat_list,Err_cov_list,p,combiner,parameters):
    sqrtErr_cov_list = [[sqrtm(Err_cov_list[l][k]) for k in range(K)] for l in range(L)]
    interf, signal, noise = compute_channel_coefficients(H_hat_list,sqrtErr_cov_list,combiner,parameters)
    SINR = compute_SINR(interf,signal,noise,p) 
    R = log2(1+np.array(SINR))
    return R

def compute_ergodic_rates(H_list,H_hat_list,p,combiner,parameters):
    ''' Optimistic ergodic rates '''
    N_sim = len(H_hat_list) 
    R = np.zeros(K)
    for n in range(N_sim):
        # Compute combiners
        H_hat = H_hat_list[n] 
        V = combiner(H_hat,parameters)
        # Equivalent channel
        H = H_list[n] 
        H_eq = np.zeros((K,K),dtype=complex)
        # Updated statistics
        for l in range(L):
            H_eq +=  herm(H[l]) @ V[l] 
            noise = np.square(norm(V[l],axis=0))    
        interf = np.square(np.abs(H_eq))
        signal = np.square(np.abs(diag(H_eq)))
        interf = interf-np.diag(signal)
        SINR_inst = compute_SINR(interf,signal,noise,p)
        R += log2(1+np.array(SINR_inst))/N_sim
    return R

def compute_channel_coefficients(H_hat_list,sqrtErr_cov_list,combiner,parameters):
    ''' Numerically evaluate the equivalent channel coefficients for the UatF bound.
    ''' 
    interf = np.zeros((K,K))
    signal = np.zeros(K,dtype=complex)
    noise = np.zeros(K)  
    N_sim = len(H_hat_list)  
    for n in range(N_sim):
        # Compute combiners
        H_hat = H_hat_list[n] 
        V = combiner(H_hat,parameters)
        # Equivalent channel
        H_eq = np.zeros((K,K),dtype=complex)
        CSI_noise = np.zeros((K,K))
        # Update expectations
        for l in range(L):
            H_eq +=  herm(H_hat[l]) @ V[l]  
            noise += np.square(norm(V[l],axis=0))/N_sim 
            sqrtErr = np.array(sqrtErr_cov_list[l])     # (K, N, N)
            tmp = sqrtErr @ V[l]                       # (K, N, K)
            CSI_noise += np.sum(np.abs(tmp)**2, axis=1) / N_sim  # (K, K)   
        interf += np.square(np.abs(H_eq))/N_sim
        interf += CSI_noise
        signal += diag(H_eq)/N_sim
    signal = np.square(np.abs(signal))  
    interf = interf-np.diag(signal)
    return  interf, signal, noise

def MMSE(H_hat,parameters):
    """ Centralized MMSE combining
    """
    UE_clusters = parameters[0]
    Psi_list = parameters[1]
    p = parameters[2]
    Q = len(UE_clusters[0])
    V_list = [np.zeros((N,K),dtype=complex) for _ in range(L)]
    for k in range(K):
        H_hat_k = np.zeros((Q*N,K),dtype=complex)
        Psi_k = np.zeros((Q*N,Q*N),dtype=complex)
        for q in range(Q):
            l = UE_clusters[k][q]
            H_hat_k[q*N:(q+1)*N,:] = H_hat[l]
            Psi_k[q*N:(q+1)*N,q*N:(q+1)*N] = Psi_list[l]
        V_MMSE = inv(H_hat_k @ diag(p) @ herm(H_hat_k) + Psi_k + eye(Q*N)) @ H_hat_k @ diag(sqrt(p))
        for q in range(Q):
            l = UE_clusters[k][q]
            V_list[l][:,k] = V_MMSE[q*N:(q+1)*N,k]
    return V_list     

def compute_parameters_MMSE(Err_cov_list,p):
    """ Computation of statistical parameters for MMSE combining. 
    """
    # Compute aggregate error covariance per access point
    Psi_list = [np.zeros((N,N),dtype=complex) for _ in range(L)] 
    for l in range(L):
        for k in range(K):
            Psi_list[l] = Psi_list[l] + p[k] * Err_cov_list[l][k]
    return Psi_list

def LTMMSE(H_hat,parameters):
    """ Local TMMSE combining
    """
    W_list = parameters[0]
    Psi_list = parameters[1]
    p = parameters[2]
    V_list = [] 
    for l in range(L):
        H_hat_l = H_hat[l]
        V_LMMSE = pinv(H_hat_l @ diag(p) @ herm(H_hat_l) + Psi_list[l] + eye(N)) @ H_hat_l @ diag(sqrt(p)) 
        V_list.append(V_LMMSE @ W_list[l]) 
    return V_list 

def compute_parameters_LTMMSE(H_hat_list,Err_cov_list,clusters,p):
    """ Computation of statistical parameters for LTMMSE combining. 
    """
    Q = len(clusters[0][0])

    # Compute aggregate error covariance per access point
    Psi_list = [np.zeros((N,N),dtype=complex) for _ in range(L)] 
    for l in range(L):
        for k in range(K):
            Psi_list[l] = Psi_list[l] + p[k] * Err_cov_list[l][k]

    # Estimate regularized and normalized covariance matrices
    UE_clusters = clusters[0]
    Pi_list = [] 
    for l in range(L):
        Pi_l = np.zeros((K,K),dtype=complex)
        for n in range(N_sim):
            H =  H_hat_list[n][l] @ diag(sqrt(p))
            Pi_l += herm(H) @ inv(H @ herm(H) + Psi_list[l] + eye(N)) @ H 
        Pi_list.append(Pi_l / N_sim)

    # Compute large scale fading decoding coefficients 
    # Initialize some useful variables
    C_list = [np.zeros((K,K),dtype=complex) for _ in range(L)]
    B = np.zeros((K*Q,K), dtype = complex)
    for l in range(Q):
        B[K*l:K*(l+1),:] = np.eye(K,dtype=complex)
    # Solve system for every UE k
    for k in range(K):
        A = np.zeros((K*Q,K*Q),dtype=complex)
        for l in range(Q):
            for j in range(Q):
                if j == l:
                    A[K*l:K*(l+1),K*l:K*(l+1)] = np.eye(K,dtype=complex)
                else:
                    A[K*l:K*(l+1),K*j:K*(j+1)] = Pi_list[UE_clusters[k][j]]
        c_k = inv(A) @ B[:,k]
        # Write coefficient onto C_list
        for l in range(Q):
            i = UE_clusters[k][l] 
            C_list[i][:,k] = c_k[K*l:K*(l+1)]
    return C_list, Psi_list


def generate_setup(plot = False):
    ''' Squared service area of length squareLength. Grid of APs, uniformely distributed APs. 
        set plot = True for visualizing the setup.'''
    sqrtL = int(sqrt(L)) 
    # Positions of APs (in meters, Cartesian coordinates)
    x = np.linspace(0,squareLength,sqrtL+1)   # "cell" boundaries on the x axis
    y = np.linspace(0,squareLength,sqrtL+1)   # "cell" boundaries on the y axis 
    x_AP = (x[:-1]+x[1:])/2
    y_AP = (y[:-1]+y[1:])/2
    pos_APs = []
    for i in range(sqrtL):
        for j in range(sqrtL):
            pos = np.array([x_AP[i],y_AP[j]])
            pos_APs.append(pos)
    # Position of UEs (in meters, Cartesian coordinates)
    pos_UEs = [squareLength*np.random.rand(2) for _ in range(K)]
    # Plot 
    if plot == True:
        plt.figure()
        fontSize = 15
        for k in range(K):
            if k == 0:
                plt.plot(pos_UEs[k][0],pos_UEs[k][1],'rv',label="UE")
            else:
                plt.plot(pos_UEs[k][0],pos_UEs[k][1],'rv')
        for l in range(L):
            if l == 0:
                plt.plot(pos_APs[l][0],pos_APs[l][1],'ob',label="AP")
            else:
                plt.plot(pos_APs[l][0],pos_APs[l][1],'ob')
        plt.xlabel('x [m]',fontsize = fontSize)
        plt.ylabel('y [m]',fontsize = fontSize)
        plt.axis('square')
        plt.xlim(0,squareLength)
        plt.ylim(0,squareLength)
        plt.xticks(x,fontsize = fontSize-2)
        plt.yticks(y,fontsize = fontSize-2)
        plt.legend()
        plt.grid()
        
        plt.show()
    pos = (pos_UEs,pos_APs,)
    return pos

def compute_ch_statistics(pos):
    """ Path loss model: 
        "Further advancements for E-UTRA physical layer aspects (Release 9)." 3GPP TS 36.814, Mar. 2017.
        3GPP UMi NLOS, exagonal layout, table B.1.2.1-1 and B.1.2.2.1-4.
        Same as Demir, Björnson and Sanguinetti, "Foundations of User-Centric Cell-Free Massive MIMO," 2021.
    """ 
    # Carrier frequency (GHz)
    f_c = 2  
    # Pathloss exponent
    PL_exp = 3.67  
    # Average channel gain in dB at a reference distance of 1 meter 
    Gamma_const = -22.7 
    # Minimum distance (i.e., the difference in height) between AP and UE in meters
    d_min = 10
    # Shadow fading std deviation [dB]
    sig_SF = 4
    # Bandwidth [Hz]
    B = 20*10**6
    # Noise figure [dB]
    noiseFigure = 7
    # Noise power [dBm]
    N0 = -174 + 10*log10(B) + noiseFigure
    # # Angular spread [rad]
    # ang_spread = 5/360*2*np.pi
    
    # UEs-APs distances (including d_min height difference), used for path loss computation
    pos_UEs = pos[0]
    pos_APs = pos[1]
    dist = np.zeros((K,L))
    for k in range(K):
        for l in range(L):
                dist[k,l]= np.sqrt(norm(pos_UEs[k]-pos_APs[l])**2 + d_min**2)

    # Uncorrelated shadow fading
    SF = sig_SF*np.random.randn(K,L)
    # UEs-UEs distances, used for computating shadow fading covariance matrices
    delta = np.zeros((K,K))
    for k in range(K):
        for j in range(K):
            delta[k,j] = norm(pos_UEs[k]-pos_UEs[j])
    # Correlated shadow fading (for each AP)
    for l in range(L):
        CovSF = np.power(2,-delta/9)  # SF covariance matrix (normalized by SF variance)
        SF[:,l] = sqrtm(CovSF) @ SF[:,l]

    # Channel gain (normalized by noise power)
    GammadB = Gamma_const - PL_exp*10*log10(dist) - 26*log10(f_c) - SF - N0
    Gamma = 10**(GammadB/10)
    return Gamma

def compute_clusters(Gamma,Q):
    """ User-centric cooperation clusters. Each user is served by the Q strongest APs.
    """
    Gamma_sorted_indexes = np.argsort(Gamma,axis=1)
    UE_clusters = Gamma_sorted_indexes[:,-Q:]
    AP_clusters = [[] for _ in range(L)]
    for k in range(K):
        for q in range(Q):
            AP_clusters[UE_clusters[k,q]].append(k)
    return UE_clusters, AP_clusters 

def draw_channel_realizations(Gamma):
    """ i.i.d. rayleigh fading"""
    H_list = []
    for _ in range(N_sim):
        H = []
        for l in range(L):
            H_iid = complex_normal(N,K)
            H.append(H_iid @ diag(sqrt(Gamma[:,l])) )
        H_list.append(H)
    return H_list

def draw_CSI_realizations(H_list,clusters):
    AP_clusters = clusters[1]
    # initialize list of channel estimates
    H_hat_list = []
    # loop over each channel relization
    for n in range(N_sim):
        H = H_list[n]
        # initialize channel estimates with channel means
        H_hat = [np.zeros((N,K),dtype=complex) for _ in range(L)]
        # store instantaneous realization if UE k is served by AP l
        for l in range(L):
            for k in AP_clusters[l]: 
                H_hat[l][:,k]= H[l][:,k]
        H_hat_list.append(H_hat)
    return H_hat_list

def compute_error_covariances(Gamma,clusters):
    AP_clusters = clusters[1]
    # initialize error covariances with zeros
    Err_cov_list = [[np.zeros((N,N),dtype=complex) for _ in range(K)] for _ in range(L)]
    # loop over access points
    for l in range(L):
        # loop over users that are not served by AP l
        for k in range(K):
            if k not in AP_clusters[l]:
                Err_cov_list[l][k] = Gamma[k,l]*eye(N,dtype=complex)
    return Err_cov_list

def complex_normal(Ni,Nj):
    return 1/sqrt(2)*(np.random.standard_normal((Ni,Nj))+1j*np.random.standard_normal((Ni,Nj))) 

def herm(x):
    return x.conj().T

if __name__ == '__main__':
    test()