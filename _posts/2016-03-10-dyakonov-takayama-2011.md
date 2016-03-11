---
layout: page
title: "Dyakonov - Takayama 2011"
category: tut
date: 2016-03-10 20:37:32
---
This example repeats the results reported in _"Dyakonov surface wave resonant transmission"_ by O. Takayama et. al., 2011 [doi:10.1364/OE.19.006339](http://dx.doi.org/10.1364/OE.19.006339).

This examples seeks to repeat the graph shown in **Figure 2b.** of the paper.

The modeled system consists of 4 layers
    1. a SF11 prism
    2. a uniaxial E7 liquid crystal
    3. another E7 liquid crystal, whose optic axis is perpendiular to the first crystal's axis
    4. a second SF11 prism

#### Imports

```python
import numpy as np
import matplotlib.pyplot as plt
from PyATMM.uniaxialTransferMatrix import *
from PyATMM.isotropicTransferMatrix import *
from PyATMM.transferMatrix import solve_transfer_matrix
```

#### Setup

```python
c = 299792458.  # m/c
w = c

# Sweep parameters
aoi = np.linspace(66, 63, 200, endpoint=True)
k_par_angles = np.linspace(np.deg2rad(40), np.deg2rad(50), 200)

# Layer 0, "prism"
n_prism = 1.77862
eps_prism = n_prism**2
k_par = np.sqrt(eps_prism) * (w/c) * np.sin(np.deg2rad(aoi))

# Layer 1, unixaial, E7
n_o = 1.520
n_e = 1.725
eps_o = n_o**2
eps_e = n_e**2
axis_1 = [1., 0., 0.]
d = 2*np.pi * 4

# Layer 2, uniaxial, E7
axis_2 = [0., 1., 0.]

# Layer 4, "prism"
```

* Since everything is defined in terms of wavelength, we set the frequency of the incident radiation, ```w```, equal to ```c```
* ```aoi``` defines the range of angles of incidence for which we do the simulation
* ```k_par_angles``` defines the range of angles of the incident radition's k-vector's projections onto the surface of the structure (relative to the x axis)
* the two E7 crystals' optic axes are aligned with the x and y axes of the simulation.
* the ticknesses of the E7 crystals are set to 4 wavelengths.


#### Calcualations

```python
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
```

The calculation is done with two loops: the outer loop is for the angle of incidence sweep (recalculated into wavevector projection), and the inner loop is for the angle of the k-vector projection sweep.
For each AOI and projection angle,  the transfer matrix is built as follows:

1. The first prism is modeled by one isotropic dynamic matrix, ```D_prism```
2. The first E7 crystal is modeled by two matrices - one uniaxial dynamic matrix, ```D_E7``` and one uniaxial propagation matrix, ```D_E7_p```.
3. The second E7 crystal is modeled by two matrices - one uniaxial dynamic matrix, ```D_E7_2``` and one uniaxial propagation matrix, ```D_E7_2p```.
4. The second prism is modeled by one isotropic dynamic matrix, ```D_prism_2```

Since the incident light comes from inside the prism, we only need to model the interface between the prism and the first E7. Both E7s are finite layers, they need to have both interfaces and internal propagation defined. The second prism is also considered to be semi-infinite, meaning that only one interface needs to be defined.

Having defined all the constituent matrices, the whole transfer matrix for the system is built by multiplying (from left to right)

1. the inverse of ```D_prism```
2. ```D_E7```
3. ```D_E7_p```
4. the inverse of ```D_E7```
5. ```D_E7_2```
6. ```D_E7_2p```
7. the inverse of ```D_E7_2```
8. ```D_prism_2```

Where 1. and 2. create the first interface (prism-E7), 3. is propagation inside the first E7, 4. and 5. create the second interface (E7-E7), 6. is propagation inside the second E7, 7. and 8. create the third interface (E7-prism).

# Plotting

```python
plt.figure(figsize=(9, 6))
plt.imshow(a_refl_ss, interpolation='none',
           extent=[np.min(k_par_angles), np.max(k_par_angles),
                   np.min(aoi), np.max(aoi)],
           aspect="auto", cmap=plt.get_cmap("afmhot"))
plt.colorbar()
plt.xlabel("K$_{par}$ Angle, radians")
plt.ticklabel_format(style='sci', useOffset=False)
plt.ylabel("AOI, degrees")
plt.title("R_ss")
plt.show()
```
