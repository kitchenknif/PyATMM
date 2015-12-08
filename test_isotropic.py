__author__ = 'Pavel Dmitriev'

import numpy as np
import matplotlib.pyplot as plt
from isotropicTransferMatrix import *
from transferMatrix import solve_transfer_matrix
import pytmm.transferMatrix as tm

c = 299792458  # m/c
w = (1/(2*np.pi))*500 * 10 ** 12
d = 5 * 10**-9

eps = 10
kx = 0
ky = 0

print(build_isotropic_layer_matrix(eps, w, kx, ky, d))
print(tm.TransferMatrix.boundingLayer(1, numpy.sqrt(10)).matrix)

exit()
ran = np.linspace(300, 1000, 3000, endpoint=False)

ps = []
pp = []
ss = []
sp = []
p = []
sum = []
for w in ran:
    D = build_isotropic_layer_matrix(eps, w*10**12, kx, ky, d)
    r_ss, r_sp, r_ps, r_pp, t_ss, t_sp, t_ps, t_pp = solve_transfer_matrix(D)

    ps.append(np.abs(r_ps**2))
    pp.append(np.abs(r_pp**2))
    ss.append(np.abs(r_ss**2))
    sp.append(np.abs(r_sp**2))

    p.append(pp[-1] + ps[-1])

    sum.append(np.abs(r_ps**2) + np.abs(r_pp**2) + np.abs(t_ps**2) + np.abs(t_pp**2))

#plt.plot(ran, sp)
#plt.plot(ran, ps)

plt.plot(ran, ss)
plt.plot(ran, pp)

plt.plot(ran, sum)
plt.legend(['ss', 'pp', 'sum'], loc='best')
#plt.legend(['ps', 'pp', 'p', 'sum'], loc='best')
plt.show(block=True)