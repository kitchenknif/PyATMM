__author__ = 'Pavel Dmitriev'

import numpy as np
import matplotlib.pyplot as plt
from uniaxialTransferMatrix import *
from transferMatrix import solve_transfer_matrix

c = 299792458  # m/c
w = (1/(2*np.pi))*500 * 10 ** 12
d = 5 * 10**-9

e_o = 10
e_e = 15
kx = 0
ky = 0

opticAxis = [0., 1., 0.]

print(build_uniaxial_layer_matrix(e_o, e_e, w, kx, ky, d, opticAxis))

exit()
ran = np.linspace(0, w, 100, endpoint=False)

ps = []
pp = []
p = []
sum = []
for kx in ran:
    D = build_uniaxial_layer_matrix(e_o, e_e, w, kx, ky, d, opticAxis)
    r_ss, r_sp, r_ps, r_pp, t_ss, t_sp, t_ps, t_pp = solve_transfer_matrix(D)

    ps.append(np.abs(r_ps**2))
    pp.append(np.abs(r_pp**2))
    p.append(pp[-1] + ps[-1])

    sum.append(np.abs(r_ps**2) + np.abs(r_pp**2) + np.abs(t_ps**2) + np.abs(t_pp**2))


#plt.plot(ran, ss)
#plt.plot(ran, sp)
plt.plot(ran, ps)
#plt.plot(ran, sp)
plt.plot(ran, pp)
plt.plot(ran, p)
plt.plot(ran, sum)
#plt.legend(['ss', 'sp', 'ps', 'pp'], loc='best')
plt.legend(['ps', 'pp', 'p', 'sum'], loc='best')
plt.show(block=True)