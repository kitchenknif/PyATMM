__author__ = 'Pavel Dmitriev'

import numpy as np
import matplotlib.pyplot as plt
from transferMatrix import *

#kx = 0.01
ky = 0.02

e0_1 = 1.02
e0_2 = 1.01
e0_3 = 1

e1 = 2.02
e2 = 2.01
e3 = 2

# theta=np.pi*3/5
# phi=np.pi/4
# psi=np.pi*7/4
theta = 0
phi = 0
psi = 0

w = 1
d = 50


ran = np.linspace(0.01, w, 100)

ss = []
sp = []
ps = []
pp = []
sum = []
for kx in ran:
    D0 = build_anisotropic_layer_matrix(e1=e0_1, e2=e0_2, e3=e0_3, theta=theta, phi=phi, psi=psi, w=w, kx=kx, ky=ky, d=d)
    D = build_anisotropic_layer_matrix(e1=e1, e2=e2, e3=e3, theta=theta, phi=phi, psi=psi, w=w, kx=kx, ky=ky, d=d)
    print("Matrix:", np.dot(np.linalg.inv(D0), D))
    r_ss, r_sp, r_ps, r_pp, t_ss, t_sp, t_ps, t_pp = solve_transfer_matrix(np.dot(np.linalg.inv(D0), D))
    ss.append(np.abs(r_ss**2))
    sp.append(np.abs(r_sp**2))
    ps.append(np.abs(r_ps**2))
    pp.append(np.abs(r_pp**2))
    sum.append(np.abs(r_ss**2) + np.abs(r_sp**2) + np.abs(r_ps**2) + np.abs(r_pp**2)
               + np.abs(t_ss**2) + np.abs(t_sp**2) + np.abs(t_ps**2) + np.abs(t_pp**2))


plt.plot(ran, ss)
plt.plot(ran, sp)
plt.plot(ran, ps)
plt.plot(ran, sp)
#plt.plot(ran, sum)
plt.legend(['ss', 'sp', 'ps', 'pp'], loc='best')
plt.show(block=True)