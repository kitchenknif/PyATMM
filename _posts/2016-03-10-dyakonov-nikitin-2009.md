---
layout: page
title: "Dyakonov - Nikitin  2009"
category: tut
date: 2016-03-10 20:38:14
---

```python
import numpy as np
import matplotlib.pyplot as plt

from PyATMM.uniaxialTransferMatrix import *
from PyATMM.isotropicTransferMatrix import *
from PyATMM.transferMatrix import solve_transfer_matrix

#
# Setup
#

c = 299792458.  # m/c
w = c

#
# Theta sweep
#
theta = np.linspace(54.20, 54.22, 50, endpoint=True)

#
# Start layer, ZnSe
#
eps_ZnSe = 6.6995
ran_kx = np.sqrt(eps_ZnSe) * (w/c) * np.sin(np.deg2rad(theta))
ky = 0

#
# Second layer, Ta_2O_5
#
eps_Ta2O5 = 4.41
ran_d = np.linspace(5, 0, 50, endpoint=True)

#
# Anisotropic layer, YV04
#
eps_YVO4_per = 3.9733
eps_YVO4_par = 4.898

phi = np.deg2rad(46.25)


a_refl_pp = []
a_refl_ss = []

a_refl_ps = []
a_refl_sp = []
for d in ran_d:
    a_refl_pp_ = []
    a_refl_ss_ = []

    a_refl_ps_ = []
    a_refl_sp_ = []

    d *= 2.*np.pi

    for kx in ran_kx:
        # Layer 0
        D_ZnSe = build_isotropic_dynamic_matrix(eps_ZnSe, w, kx, ky)

        # Layer 1
        D_Ta2O5 = build_isotropic_dynamic_matrix(eps_Ta2O5, w, kx, ky)
        D_Ta2O5_p = build_isotropic_propagation_matrix(eps_Ta2O5, w, kx, ky, d)

        # Layer 2
        D_YVo4 = build_uniaxial_dynamic_matrix(eps_YVO4_per, eps_YVO4_par, w, kx, ky, opticAxis=[np.cos(phi), np.sin(phi), 0])

        # Multiply Matricies
        D = np.dot(np.dot(np.dot(np.linalg.inv(D_ZnSe), D_Ta2O5), D_Ta2O5_p), np.dot(np.linalg.inv(D_Ta2O5), D_YVo4))

        # Solve Matrix
        r_ss, r_sp, r_ps, r_pp, t_ss, t_sp, t_ps, t_pp = solve_transfer_matrix(D)
        a_refl_pp_.append(np.abs(r_pp**2))
        a_refl_ss_.append(np.abs(r_ss**2))

        a_refl_sp_.append(np.abs(r_sp**2))
        a_refl_ps_.append(np.abs(r_ps**2))

    a_refl_pp.append(a_refl_pp_)
    a_refl_ss.append(a_refl_ss_)

    a_refl_sp.append(a_refl_sp_)
    a_refl_ps.append(a_refl_ps_)


plt.figure(figsize=(9, 6))
plt.imshow(a_refl_ps, interpolation='none', extent=[np.min(theta), np.max(theta), np.min(ran_d), np.max(ran_d)],
           aspect="auto",  cmap=plt.get_cmap("jet"))

plt.colorbar()
plt.xlabel("$\Theta$, degrees")
plt.xticks([54.20, 54.205, 54.21, 54.215, 54.22])
plt.ticklabel_format(style = 'sci', useOffset=False)
plt.ylabel("$\Delta / \lambda$")
plt.title("R_ps")

plt.show(block=True)
```
