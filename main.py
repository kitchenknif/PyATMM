__author__ = 'Pavel Dmitriev'

from transferMatrix import *

pi = 0
A = build_anisotropic_layer_matrix(1.5, 1.5, 1.5, pi/2, pi/2, 0, 1, 0, 0, 200*10**(-6) )

print(A)

