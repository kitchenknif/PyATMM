import numpy as np
from transferMatrix import *
import scipy
from scipy import linalg, matrix

def null(A, eps=1e-15):
    u, s, vh = scipy.linalg.svd(A)
    null_mask = (s <= eps)
    null_space = scipy.compress(null_mask, vh, axis=0)
    return scipy.transpose(null_space)

def build_matrix(kx, ky, kz, Eps, w):
    c = 1 # m/c
    return (w**2 / c**2 ) * Eps - numpy.dot([kx, ky, kz], [kx, ky, kz])*numpy.eye(3) + numpy.outer([kx, ky, kz], [kx, ky, kz])


kx=0.5
ky=0.5

pi = 3.14
P_an, gamma_an, Eps = build_anisotropic_layer_matrix(e1=1, e2=1, e3=1, theta=pi/2, phi=pi/4, psi=0, w=1, kx=kx, ky=ky, d=200*10**(-6) )
print('gamma 0', gamma_an[0])

A = build_matrix(kx=kx, ky=ky, kz=gamma_an[0], Eps=Eps, w=1)
#print(np.linalg.det(A))

A[0,0] = 0
A[0,1] = 0
A[0,2] = 0
print('A', A)

P_num = null(A).T[0]

print("An", P_an[0])
print("Num", P_num)


print(numpy.dot(numpy.dot(Eps, P_num), [kx, ky, gamma_an[0]]))

print(numpy.dot(numpy.dot(Eps, P_an[0]), [kx, ky, gamma_an[0]]))

# print(np.dot(A, P_num))
#
# A = build_matrix(kx=kx, ky=ky, kz=gamma_an[0], Eps=Eps, w=1)
# print(np.dot(A, P_an[0]))