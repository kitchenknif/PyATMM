__author__ = 'Pavel Dmitriev'

import numpy as np
import matplotlib.pyplot as plt
from uniaxialTransferMatrix import *
from transferMatrix import solve_transfer_matrix
import PyTMM.pytmm.transferMatrix as tm
import scipy.linalg

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

# d_nm = 800 / (n_2 * 4) # quarter-wavelength coating
d_nm = 2000
d_m = d_nm * 10**-9



refl_p = []
refl_s = []
for i in ran_l:
    B = tm.TransferMatrix.layer(n_1, d_nm, i*10**9, 0, tm.Polarization.s)
    C = tm.TransferMatrix.layer(n_2, d_nm, i*10**9, 0, tm.Polarization.p)
    M = scipy.linalg.block_diag(B.matrix, C.matrix)
    r_ss, r_sp, r_ps, r_pp, t_ss, t_sp, t_ps, t_pp = solve_transfer_matrix(M)
    refl_p.append(np.abs(r_pp**2))
    refl_s.append(np.abs(r_ss**2))


a_refl_p = []
a_refl_s = []
for w in ran_w:
    #D = build_isotropic_bounding_layer_matrix(eps_1, w, kx, ky)
    D = build_uniaxial_layer_matrix(eps_1, eps_2, w, kx, ky, d_m, opticAxis=([0., 1., 0.]))
    #D = build_isotropic_propagation_matrix(eps_1, w, kx, ky, d_m)

    #r_ss, r_sp, r_ps, r_pp, t_ss, t_sp, t_ps, t_pp = solve_transfer_matrix(numpy.dot(D_b, D))
    #a_refl.append(np.abs(r_pp**2))

    r_ss, r_sp, r_ps, r_pp, t_ss, t_sp, t_ps, t_pp = solve_transfer_matrix(D)
    a_refl_p.append(np.abs(r_pp**2))
    a_refl_s.append(np.abs(r_ss**2))


#PyATMM
plt.plot(ran_l*10**9, a_refl_p)
plt.plot(ran_l*10**9, a_refl_s)
#plt.plot(ran_l*10**9, sum)

#PyTMM
plt.plot(ran_l*10**9, refl_p, 'o')
plt.plot(ran_l*10**9, refl_s, 'o')



plt.xlabel("Wavelength, nm")
plt.ylabel("Reflectance")
plt.title("Reflectance of ideal single-layer antireflective coating")
plt.legend(['PP', 'SS'], loc='best')
plt.show(block=True)