import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg

import PyTMM.transferMatrix as tm

from PyATMM.isotropicTransferMatrix import *
from PyATMM.transferMatrix import solve_transfer_matrix

__author__ = 'Pavel Dmitriev'

#
# Setup
#
c = 299792458  # m/c

w = 500 * 10 **12
l = 2*np.pi*c / w
angle = np.linspace(0, 90, 200, endpoint=False)
ran_kx = (w/c) * np.sin(np.deg2rad(angle))

eps_1 = 2.25
n_1 = np.sqrt(eps_1)

ky = 0


#
# PyTMM
#
refl_p = []
refl_s = []
for kx in ran_kx:
    B = tm.TransferMatrix.boundingLayer(1, n_1, np.arcsin(kx*c/w), tm.Polarization.s)
    C = tm.TransferMatrix.boundingLayer(1, n_1, np.arcsin(kx*c/w), tm.Polarization.p)

    M = scipy.linalg.block_diag(B.matrix, C.matrix)
    r_ss, r_sp, r_ps, r_pp, t_ss, t_sp, t_ps, t_pp = solve_transfer_matrix(M)
    refl_p.append(np.abs(r_pp**2))
    refl_s.append(np.abs(r_ss**2))

#
# PyATMM
#
a_refl_p = []
a_refl_s = []
for kx in ran_kx:
    D = build_isotropic_bounding_layer_matrix(eps_1, w, kx, ky)

    r_ss, r_sp, r_ps, r_pp, t_ss, t_sp, t_ps, t_pp = solve_transfer_matrix(D)
    a_refl_p.append(np.abs(r_pp**2))
    a_refl_s.append(np.abs(r_ss**2))
    assert np.isclose(r_sp, 0)
    assert np.isclose(r_ps, 0)

#PyATMM
plt.plot(angle, a_refl_p)
plt.plot(angle, a_refl_s)

#PyTMM
plt.plot(angle, refl_p, '+')
plt.plot(angle, refl_s, '+')


plt.xlabel("kx, $m^{-1}$")
plt.ylabel("Reflectance")
plt.title("Brewster angle")
plt.legend(['PP', 'SS', 'P', 'S'], loc='best')
plt.show(block=True)