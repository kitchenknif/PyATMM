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

eps_1 = 6.7
n_1 = np.sqrt(eps_1)

eps_2 = 4.41
n_2 = np.sqrt(eps_2)

ky = 0


#
# PyTMM
#
refl_p = []
refl_s = []
for kx in ran_kx:
    B = tm.TransferMatrix.boundingLayer(n_1, n_2, np.arcsin(kx*c/w), tm.Polarization.s)
    C = tm.TransferMatrix.boundingLayer(n_1, n_2, np.arcsin(kx*c/w), tm.Polarization.p)
    #a.appendRight(b)

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
    D_0 = build_isotropic_dynamic_matrix(eps_1, w, n_1*kx, ky)
    D_1 = build_isotropic_dynamic_matrix(eps_2, w, n_1*kx, ky)
    D = np.dot(np.linalg.inv(D_0), D_1)
    r_ss, r_sp, r_ps, r_pp, t_ss, t_sp, t_ps, t_pp = solve_transfer_matrix(D)
    a_refl_p.append(np.abs(r_pp**2))
    a_refl_s.append(np.abs(r_ss**2))

#PyTMM
plt.plot(angle, refl_p, '+')
plt.plot(angle, refl_s, '+')

#PyATMM
plt.plot(angle, a_refl_p)
plt.plot(angle, a_refl_s)

plt.xlabel("angle, degrees")
plt.ylabel("Reflectance")
plt.title("Total internal reflection")
plt.legend(['PP', 'SS', 'P', 'S'], loc='best')
plt.show(block=True)