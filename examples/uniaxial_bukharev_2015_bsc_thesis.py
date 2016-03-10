import numpy as np
import matplotlib.pyplot as plt
from PyATMM.uniaxialTransferMatrix import *
from PyATMM.isotropicTransferMatrix import *
from PyATMM.transferMatrix import solve_transfer_matrix

__author__ = 'Pavel Dmitriev'

#
# Setup
#
c = 299792458  # m/c

w = 500 * 10 **12
l = 2*np.pi*c / w

angle = np.linspace(0, 90, 500, endpoint=False)
ran_kx = (w/c) * np.sin(np.deg2rad(angle))


n_g = 1.5
eps_g = n_g**2

n_ord = 1
eps_ord = n_ord**2

n_ex = 1.2
eps_ex = n_ex**2

ky = 0


a_refl_pp = []
a_refl_ss = []

a_refl_ps = []
a_refl_sp = []

for kx in ran_kx:
    D_0 = build_isotropic_dynamic_matrix(eps_g, w, n_g*kx, ky)
    D_1 = build_uniaxial_dynamic_matrix(eps_ord, eps_ex, w, n_g*kx, ky, opticAxis=([0, 1./np.sqrt(2), 1./np.sqrt(2)]))
    D = np.dot(np.linalg.inv(D_0), D_1)

    r_ss, r_sp, r_ps, r_pp, t_ss, t_sp, t_ps, t_pp = solve_transfer_matrix(D)
    a_refl_pp.append(np.abs(r_pp**2))
    a_refl_ss.append(np.abs(r_ss**2))

    a_refl_sp.append(np.abs(r_sp**2))
    a_refl_ps.append(np.abs(r_ps**2))

#PyATMM

#TE
plt.plot(angle, a_refl_ss)
plt.plot(angle, a_refl_sp)
plt.plot(angle, numpy.add(a_refl_ss, a_refl_sp))
plt.legend(['SS', 'SP', 'Sum'], loc='best')

#TM
#plt.plot(angle, a_refl_pp)
#plt.plot(angle, a_refl_ps)
#plt.plot(angle, numpy.add(a_refl_pp, a_refl_ps))
#plt.legend(['PP', 'PS', 'Sum'], loc='best')

plt.xlabel("angle, degrees")
plt.ylabel("Reflectance")
plt.title("Uniaxial total internal reflection")
plt.show(block=True)