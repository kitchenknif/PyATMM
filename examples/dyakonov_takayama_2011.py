import numpy as np
import matplotlib.pyplot as plt
from PyATMM.uniaxialTransferMatrix import *
from PyATMM.isotropicTransferMatrix import *
from PyATMM.transferMatrix import solve_transfer_matrix

__author__ = 'Pavel Dmitriev'

c = 299792458.  # m/c
w = c
# w = 2*np.pi * 500. * 10 **12


#
# Sweep parameters
#
aoi = np.linspace(66, 63, 200, endpoint=True)
k_par_angles = np.linspace(np.deg2rad(40), np.deg2rad(50), 200)


#
# Layer 0, "prism"
#
n_prism = 1.77862
eps_prism = n_prism**2
k_par = np.sqrt(eps_prism) * (w/c) * np.sin(np.deg2rad(aoi))

#
# Layer 1, unixaial, E7
#
n_o = 1.520
n_e = 1.725
eps_o = n_o**2
eps_e = n_e**2
axis_1 = [1., 0., 0.]
d = 2*np.pi * 4

#
# Layer 2, uniaxial, E7
#
axis_2 = [0., 1., 0.]

#
# Layer 4, "prism"
#


a_refl_pp = []
a_refl_ss = []

a_refl_ps = []
a_refl_sp = []


for k in k_par:
    a_refl_pp_ = []
    a_refl_ss_ = []

    a_refl_ps_ = []
    a_refl_sp_ = []

    for o in k_par_angles:
        kx = np.cos(o) * k
        ky = np.sin(o) * k

        # 0
        D_prism = build_isotropic_dynamic_matrix(eps_prism, w, kx, ky)

        # 1
        D_E7_1 = build_uniaxial_dynamic_matrix(eps_o, eps_e, w, kx, ky,
                                               opticAxis=axis_1)
        D_E7_1p = build_uniaxial_propagation_matrix(eps_o, eps_e, w, kx, ky, d,
                                                    opticAxis=axis_1)

        D_1 = np.dot(np.dot(np.linalg.inv(D_prism), D_E7_1), D_E7_1p)

        # 2
        D_E7_2 = build_uniaxial_dynamic_matrix(eps_o, eps_e, w, kx, ky,
                                               opticAxis=axis_2)
        D_E7_2p = build_uniaxial_propagation_matrix(eps_o, eps_e, w, kx, ky, d,
                                                    opticAxis=axis_2)

        D_2 = np.dot(np.dot(np.linalg.inv(D_E7_1), D_E7_2), D_E7_2p)

        # 3
        D_prism_2 = np.dot(np.linalg.inv(D_E7_2), D_prism)

        D = np.dot(D_1, np.dot(D_2, D_prism_2))

        r_ss, r_sp, r_ps, r_pp, \
        t_ss, t_sp, t_ps, t_pp = solve_transfer_matrix(D)

        a_refl_pp_.append(np.abs(r_pp**2))
        a_refl_ss_.append(np.abs(r_ss**2))

        a_refl_sp_.append(np.abs(r_sp**2))
        a_refl_ps_.append(np.abs(r_ps**2))

    a_refl_pp.append(a_refl_pp_)
    a_refl_ss.append(a_refl_ss_)

    a_refl_sp.append(a_refl_sp_)
    a_refl_ps.append(a_refl_ps_)

plt.figure(figsize=(9, 6))
plt.imshow(a_refl_ss, interpolation='none',
           extent=[np.min(k_par_angles), np.max(k_par_angles),
                   np.min(aoi), np.max(aoi)],
           aspect="auto", cmap=plt.get_cmap("afmhot"))
plt.colorbar()
plt.xlabel("K$_{par}$ Angle, radians")
plt.ticklabel_format(style='sci', useOffset=False)
plt.ylabel("AOI, degrees")
plt.title("R_ps")

# plt.suptitle("Otto Geometry test", fontsize=40)
# plt.savefig("otto_test.pdf", bbox_inches='tight')
plt.show()
