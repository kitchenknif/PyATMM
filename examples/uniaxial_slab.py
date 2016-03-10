import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg

import PyTMM.transferMatrix as tm

from PyATMM.uniaxialTransferMatrix import *
from PyATMM.transferMatrix import solve_transfer_matrix

__author__ = 'Pavel Dmitriev'

#
# Setup
#
c = 299792458  # m/c

ran_w = np.linspace(50, 1200, 100, endpoint=False)
ran_w = np.multiply(ran_w, 10 ** 12) # (1/(2*np.pi))*
ran_l = np.divide(c*(2*np.pi), ran_w)

eps_1 = 2
eps_2 = 2.25
n_1 = np.sqrt(eps_1)
n_2 = np.sqrt(eps_2)

kx = 0
ky = 0

d_nm = 2000
d_m = d_nm * 10**-9

#
# PyTMM
#
refl_p = []
refl_s = []
for i in ran_l:
    B = tm.TransferMatrix.layer(n_1, d_nm, i*10**9, 0, tm.Polarization.s)
    C = tm.TransferMatrix.layer(n_2, d_nm, i*10**9, 0, tm.Polarization.p)
    M = scipy.linalg.block_diag(B.matrix, C.matrix)
    r_ss, r_sp, r_ps, r_pp, t_ss, t_sp, t_ps, t_pp = solve_transfer_matrix(M)
    refl_p.append(np.abs(r_pp**2))
    refl_s.append(np.abs(r_ss**2))

#
# PyATMM
#
a_refl_p = []
a_refl_s = []
for w in ran_w:
    D = build_uniaxial_layer_matrix(eps_1, eps_2, w, kx, ky, d_m, opticAxis=([0., 1., 0.]))

    r_ss, r_sp, r_ps, r_pp, t_ss, t_sp, t_ps, t_pp = solve_transfer_matrix(D)
    a_refl_p.append(np.abs(r_pp**2))
    a_refl_s.append(np.abs(r_ss**2))


#PyATMM
plt.plot(ran_l*10**9, a_refl_p)
plt.plot(ran_l*10**9, a_refl_s)

#PyTMM
plt.plot(ran_l*10**9, refl_p, 'o')
plt.plot(ran_l*10**9, refl_s, 'o')


plt.xlabel("Wavelength, nm")
plt.ylabel("Reflectance")
plt.title("Reflectance uniaxial crystal")
plt.legend(['PP', 'SS'], loc='best')
plt.show(block=True)