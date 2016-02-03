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


kx=0.01
ky=0.02

e1=1.02
e2=1.01
e3=1

# theta=np.pi*3/5
# phi=np.pi/4
# psi=np.pi*7/4
theta=0
phi=0
psi=0

w=1
d=5

P_an, gamma_an, Eps = build_anisotropic_layer_matrix(e1=e1, e2=e2, e3=e3, theta=theta, phi=phi, psi=psi, w=w, kx=kx, ky=ky, d=d)

# print('gamma 0', gamma_an[0])
A = [build_matrix(kx=kx, ky=ky, kz=g, Eps=Eps, w=1) for g in gamma_an]
# print(np.linalg.det(A))

for a in A:
    a[2, 0] = 0
    a[2, 1] = 0
    a[2, 2] = 0

# print('A', A)
#
P_num = [null(a).T[0] for a in A]
#
# print("An", P_an[0])
# print("Num", P_num)
#
#
# #print(numpy.dot(numpy.dot(Eps, P_num), [kx, ky, gamma_an[0]]))
#
# print(numpy.dot(numpy.dot(Eps, P_an[0]), [kx, ky, gamma_an[0]]))

# print(np.dot(A, P_num))
#
# A = build_matrix(kx=kx, ky=ky, kz=gamma_an[0], Eps=Eps, w=1)
# print(np.dot(A, P_an[0]))
# A = build_matrix(kx=kx, ky=ky, kz=gamma_an[1], Eps=Eps, w=1)
# print(np.dot(A, P_an[1]))
# A = build_matrix(kx=kx, ky=ky, kz=gamma_an[2], Eps=Eps, w=1)
# print(np.dot(A, P_an[2]))
# A = build_matrix(kx=kx, ky=ky, kz=gamma_an[3], Eps=Eps, w=1)
# print(np.dot(A, P_an[3]))
print("Null-space")
print("Pol: ", np.abs(P_num[0]), "Kz: ", np.abs(gamma_an[0]))
print("Pol: ", np.abs(P_num[1]), "Kz: ", np.abs(gamma_an[1]))
print("Pol: ", np.abs(P_num[2]), "Kz: ", np.abs(gamma_an[2]))
print("Pol: ", np.abs(P_num[3]), "Kz: ", np.abs(gamma_an[3]))
print("Analytic")
print("Pol: ", np.abs(P_an[0]), "Kz: ", np.abs(gamma_an[0]))
print("Pol: ", np.abs(P_an[1]), "Kz: ", np.abs(gamma_an[1]))
print("Pol: ", np.abs(P_an[2]), "Kz: ", np.abs(gamma_an[2]))
print("Pol: ", np.abs(P_an[3]), "Kz: ", np.abs(gamma_an[3]))