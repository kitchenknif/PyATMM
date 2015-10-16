__author__ = 'Pavel Dmitriev'

from transferMatrix import *

pi = 3.14

A = build_anisotropic_layer_matrix(e1=1.5, e2=1.5, e3=1.5, theta=pi/2, phi=pi/4, psi=0, w=1, kx=0, ky=0, d=200*10**(-6) )

print(A)

