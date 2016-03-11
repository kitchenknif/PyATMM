---
layout: page
title: "Isotropic slab"
category: tut
date: 2016-03-10 20:36:10
---
This example models the reflectivity of a 2-micron thick isotropic slab with a refractive index of 1.5.

#### Imports

```python
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg

import PyTMM.transferMatrix as tm

from PyATMM.isotropicTransferMatrix import *
from PyATMM.transferMatrix import solve_transfer_matrix
```
* PyTMM is used for checking PyATMM's work.

#### Setup
Define all the variables used to describe the system

```python
c = 299792458  # m/c

ran_w = np.linspace(50, 1200, 100, endpoint=False)
ran_w = np.multiply(ran_w, 10**12)  # (1/(2*np.pi))*
ran_l = np.divide(c*(2*np.pi), ran_w)

eps_1 = 2.25
n_1 = np.sqrt(eps_1)

kx = 0
ky = 0

d_nm = 2000
d_m = d_nm * 10**-9
```
* c - speed of light
* ran_w - range of frequencies for which we are calculating reflectivities
* ran_l - ran_w converted to wavelengths (for PyTMM)
* n_1 - refractive index, eps_1 = n_1**2
* d_m - thickness of slab in meters (d_nm - in nanometers, for PyTMM)

#### PyTMM calcualations

```python
refl_p = []
refl_s = []
for i in ran_l:
    B = tm.TransferMatrix.layer(n_1, d_nm, i*10**9, 0, tm.Polarization.s)
    C = tm.TransferMatrix.layer(n_1, d_nm, i*10**9, 0, tm.Polarization.p)

    M = scipy.linalg.block_diag(B.matrix, C.matrix)
    r_ss, r_sp, r_ps, r_pp, t_ss, t_sp, t_ps, t_pp = solve_transfer_matrix(M)
    refl_p.append(np.abs(r_pp**2))
    refl_s.append(np.abs(r_ss**2))
```

1. Define lists to store reflectivies
2. Iterate over range of wavelengths (ran_l)
    1. Build 2x2 transfer matrices B & C for s and p polarizations (PyTMM)
    2. Build 4x4 generalized transfer matrix
    3. Solve generalized transfer matrix (PyATMM)
    4. Push reflectivities to lists

#### PyATMM

```python
a_refl_p = []
a_refl_s = []
for w in ran_w:
    D = build_isotropic_layer_matrix(eps_1, w, kx, ky, d_m)

    r_ss, r_sp, r_ps, r_pp, t_ss, t_sp, t_ps, t_pp = solve_transfer_matrix(D)
    a_refl_p.append(np.abs(r_pp**2))
    a_refl_s.append(np.abs(r_ss**2))
```

1. Define lists to store reflectivies
2. Iterate over range of frequencies (ran_w)
    1. Build 4x4 generalized transfer matrix (done inside ```build_isotropic_layer_matrix```)
        1. Build boundary layer matrix "D_0"
            1. Inverse of Dynamic matrix for vacuum (```build_isotropic_dynamic_matrix```)
            2. Dynamic matrix for slab (```build_isotropic_dynamic_matrix```)
        2. Build propagation matrix  "D_p" (```build_isotropic_propagation_matrix```)
        3. Building boundary layer matrix  "D_1"
            1. Inverse of Dynamic matrix for slab (```build_isotropic_dynamic_matrix```)
            2. Dynamic matrix for vacuum (```build_isotropic_dynamic_matrix```)
        4. Multiply matrices D_0 * D_p * D_1
    3. Solve generalized transfer matrix (PyATMM)
    4. Push reflectivities to lists

#### Plot
```python
#PyATMM
plt.plot(ran_l*10**9, a_refl_p)
plt.plot(ran_l*10**9, a_refl_s)

#PyTMM
plt.plot(ran_l*10**9, refl_p, 'o')
plt.plot(ran_l*10**9, refl_s, 'o')

plt.xlabel("Wavelength, nm")
plt.ylabel("Reflectance")
plt.title("Reflectance of a 2 micron thick glass slab")
plt.legend(['PP', 'SS', 'P', 'S'], loc='best')
plt.show(block=True)
```
