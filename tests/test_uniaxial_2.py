__author__ = 'Pavel Dmitriev'

import numpy as np
import matplotlib.pyplot as plt
from uniaxialTransferMatrix import *
from transferMatrix import solve_transfer_matrix
import scipy.linalg

c = 299792458  # m/c

w = 500 * 10 **12
l = 2*np.pi*c / w
#ran_kx = np.linspace(0, (w/c)*0.99, 100, endpoint=False)

angle = np.linspace(0, 90, 200, endpoint=False)
ran_kx = (w/c) * np.sin(np.deg2rad(angle))


n_g = 1.5
eps_g = n_g**2

n_ord = 1
eps_ord = n_ord**2

n_ex = 1.2
eps_ex = n_ex**2


d_nm = 2000
d_m = d_nm * 10**-9


ky = 0


a_refl_pp = []
a_refl_ss = []

a_refl_ps = []
a_refl_sp = []

for kx in ran_kx:
    #D =
    D = build_uniaxial_layer_matrix(eps_ord, eps_ex, w, kx, ky, d_m, opticAxis=([1/np.sqrt(2), 1/np.sqrt(2), 0.]))

    r_ss, r_sp, r_ps, r_pp, t_ss, t_sp, t_ps, t_pp = solve_transfer_matrix(D)
    a_refl_pp.append(np.abs(r_pp**2))
    a_refl_ss.append(np.abs(r_ss**2))

    a_refl_sp.append(np.abs(r_sp**2))
    a_refl_ps.append(np.abs(r_ps**2))

#PyATMM
plt.plot(angle, a_refl_pp)
#plt.plot(ran_kx, a_refl_ss)

plt.plot(angle, a_refl_ps)
#plt.plot(ran_kx, a_refl_sp)
#plt.plot(ran_l*10**9, sum)


plt.xlabel("kx, $m^{-1}$")
plt.ylabel("Reflectance")
#plt.title("Reflectance of ideal single-layer antireflective coating")
plt.legend(['PP', 'SS', 'PS', 'SP'], loc='best')
plt.show(block=True)