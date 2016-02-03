__author__ = 'Pavel Dmitriev'

import numpy as np
import matplotlib.pyplot as plt
from isotropicTransferMatrix import *
from transferMatrix import solve_transfer_matrix
import PyTMM.pytmm.transferMatrix as tm
import scipy.linalg

c = 299792458  # m/c

w = 500 * 10 **12
l = 2*np.pi*c / w
angle = np.linspace(0, 90, 200, endpoint=False)
ran_kx = (w/c) * np.sin(np.deg2rad(angle))

eps_1 = 2.25
n_1 = np.sqrt(eps_1)

ky = 0

refl_p = []
refl_s = []
for kx in ran_kx:
    B = tm.TransferMatrix.boundingLayer(1, n_1, np.arcsin(kx*c/w), tm.Polarization.s)
    C = tm.TransferMatrix.boundingLayer(1, n_1, np.arcsin(kx*c/w), tm.Polarization.p)
    #a.appendRight(b)

    M = scipy.linalg.block_diag(B.matrix, C.matrix)
    r_ss, r_sp, r_ps, r_pp, t_ss, t_sp, t_ps, t_pp = solve_transfer_matrix(M)
    refl_p.append(np.abs(r_pp**2))
    refl_s.append(np.abs(r_ss**2))

a_refl_p = []
a_refl_s = []
for kx in ran_kx:
    D = build_isotropic_bounding_layer_matrix(eps_1, w, kx, ky)
    #D = build_isotropic_layer_matrix(eps_1, w, kx, ky, d_m)
    #D = build_isotropic_propagation_matrix(eps_1, w, kx, ky, d_m)

    #r_ss, r_sp, r_ps, r_pp, t_ss, t_sp, t_ps, t_pp = solve_transfer_matrix(numpy.dot(D_b, D))
    #a_refl.append(np.abs(r_pp**2))

    r_ss, r_sp, r_ps, r_pp, t_ss, t_sp, t_ps, t_pp = solve_transfer_matrix(D)
    a_refl_p.append(np.abs(r_pp**2))
    a_refl_s.append(np.abs(r_ss**2))
    assert np.isclose(r_sp, 0)
    assert np.isclose(r_ps, 0)

#PyATMM
plt.plot(angle, a_refl_p)
plt.plot(angle, a_refl_s)
#plt.plot(ran_l*10**9, sum)

#PyTMM
#plt.plot(ran_l*10**9, refl0, 'o')
plt.plot(angle, refl_p, '+')
plt.plot(angle, refl_s, '+')


plt.xlabel("kx, $m^{-1}$")
plt.ylabel("Reflectance")
#plt.title("Reflectance of ideal single-layer antireflective coating")
plt.legend(['PP', 'SS', 'P', 'S'], loc='best')
plt.show(block=True)