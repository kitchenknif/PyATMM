---
layout: page
title: "Dyakonov - Nikitin  2009"
category: tut
date: 2016-03-10 20:38:14
---
This example repeats the results reported in _"Polarization conversion spectroscopy of hybrid modes"_ by A. Yu. Nikitin et. al., 2009 [doi:10.1364/OL.34.003911](http://dx.doi.org/10.1364/OL.34.003911).

Probably because the dielectric permittivities presented in the paper were rounded, an exact repeat of the results turned out to be difficult to achieve. This examples seeks to repeat the graph shown in **Figure 2c.** of the paper.

The modeled system consists of 3 layers
    1. a ZnSe prism
    2. a Ta2O5 isotropic spacer
    3. and a semi-infinite YV04 anisotropic layer

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

    # Angle of incidence sweep
    theta = np.linspace(54.20, 54.22, 50, endpoint=True)

    # Angle between optic axis and projection of k-vector
    phi = np.deg2rad(46.25)

    # Start layer (prism), ZnSe
    eps_ZnSe = 6.6995
    ran_kx = np.sqrt(eps_ZnSe) * (w/c) * np.sin(np.deg2rad(theta))
    ky = 0

    # Second layer, Ta_2O_5
    eps_Ta2O5 = 4.41
    ran_d = np.linspace(5, 0, 50, endpoint=True)

    # Anisotropic layer, YV04
    eps_YVO4_per = 3.9733
    eps_YVO4_par = 4.898
```

* Since everything is defined in terms of wavelength, we set the frequency of the incident radiation, ```w``` equal to ```c```
* ```theta``` defines the range of angles of incidence for which we do the simulation
* ```ran_d``` defines the range of Ta2O5 thicknesses for which we do the simulation

#### Calculations

```python
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
```

The calculation is done with two loops: the outer loop is for the spacer thickness sweep, and the inner loop is for the angle of incidence sweep (recalculated into wavevector projection).
For each spacer thickness & angle of incidence, the transfer matrix is built as follows:

1. The prism is modeled by one isotropic dynamic matrix, ```D_ZnSe```
2. The spacer is modeled by two matrices - one isotropic dynamic matrix, ```D_Ta2O5``` and one isotropic propagation matrix, ```D_Ta2O5_p```.
3. The uniaxial crystal is modeled by one uniaxial dynamic matrix, ```D_YVo4```.

Since the incident light comes from inside the prism, we only need to model the interface between the prism and the spacer. The spacer is a finite layer, so it needs to have both interfaces and internal propagation defined. The uniaxial layer is considered to be semi-infinite, meaning that only one interface needs to be defined.

Having defined all the constituent matrices, the whole transfer matrix for the system is built by multiplying (from left to right)

1. the inverse of ```D_ZnSe```
2. ```D_Ta2O5```
3. ```D_Ta2O5_p```
4. the inverse of ```D_Ta2O5```
5. ```D_YVo4```

Where 1. and 2. create the first interface (prism-spacer), 3. is propagation inside the spacer, 4. and 5. create the second interface (spacer-uniaxial crystal).

#### Plotting

```python
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
