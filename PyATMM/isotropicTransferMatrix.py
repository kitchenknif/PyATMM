__author__ = 'Pavel Dmitriev'

import numpy
import numpy.linalg
from PyATMM.transferMatrix import *


def build_isotropic_layer_matrix(eps, w, kx, ky, d):
    #D_0 = build_isotropic_dynamic_matrix(1, w, kx, ky)
    #D = build_isotropic_dynamic_matrix(eps, w, kx, ky)
    D = build_isotropic_bounding_layer_matrix(eps, w, kx, ky)
    P = build_isotropic_propagation_matrix(eps, w, kx, ky, d)

    #LayerMatrix = numpy.dot(numpy.linalg.inv(D_0), numpy.dot(numpy.dot(D, P), numpy.dot(numpy.linalg.inv(D), D_0)))

    LayerMatrix = numpy.dot(D, numpy.dot(P, numpy.linalg.inv(D)))
    return LayerMatrix

def build_isotropic_bounding_layer_matrix(eps, w, kx, ky):
    #
    # Build boundary transition matrix
    #
    D_0 = build_isotropic_dynamic_matrix(1, w, kx, ky)
    D = build_isotropic_dynamic_matrix(eps, w, kx, ky)

    return numpy.dot(numpy.linalg.inv(D_0), D)


#
# Building blocks
#
def build_isotropic_propagation_matrix(eps, w, kx, ky, d):
    mu = 1.
    c = 299792458.  # m/c

    mod_kz = numpy.sqrt(eps*(w/c)**2 - kx**2 - ky**2, dtype=numpy.complex128)
    gamma = [mod_kz, -mod_kz, mod_kz, -mod_kz]

    P = numpy.asarray(
        [
            [numpy.exp(-1j*gamma[0]*d), 0, 0, 0],
            [0, numpy.exp(-1j*gamma[1]*d), 0, 0],
            [0, 0, numpy.exp(-1j*gamma[2]*d), 0],
            [0, 0, 0, numpy.exp(-1j*gamma[3]*d)]
        ], dtype=numpy.complex128
    )
    return P


def build_isotropic_dynamic_matrix(eps, w, kx, ky):
    mu = 1.
    c = 299792458.  # m/c

    mod_kz = numpy.sqrt(eps*(w/c)**2 - kx**2 - ky**2, dtype=numpy.complex128)
    gamma = [mod_kz, -mod_kz, mod_kz, -mod_kz]
    k = [numpy.asarray([kx, ky, g]) for g in gamma]

    #
    # Build polarization vectors
    #
    p = isotropic_polarizations(w, eps, kx, ky, gamma)
    q = [(c/(w*mu))*numpy.cross(ki, pi) for ki, pi in zip(k, p)]

    #
    # Build boundary transition matrix
    #
    D = numpy.asarray(
        [
            [p[0][0], p[1][0], p[2][0], p[3][0]],
            [q[0][1], q[1][1], q[2][1], q[3][1]],
            [p[0][1], p[1][1], p[2][1], p[3][1]],
            [q[0][0], q[1][0], q[2][0], q[3][0]]
        ], dtype=numpy.complex128
    )
    return D


def isotropic_polarizations(w, eps, kx, ky, kz):
    k = [[kx, ky, ki] for ki in kz]

    # TODO
    # K-vector aligned
    # k = [kx, ky, kz]
    #
    if not numpy.isclose(kx, 0) or not numpy.isclose(ky, 0):
        p_3 = numpy.cross(k[2], [-kx, -ky, kz[2]])
        p_4 = numpy.cross(k[3], [-kx, -ky, kz[3]])

        p_1 = numpy.cross(k[0], [-kx, -ky, kz[0]])
        p_1 = numpy.cross(k[0], p_1)

        p_2 = numpy.cross(k[1], [-kx, -ky, kz[1]])
        p_2 = numpy.cross(k[1], p_2)
    else:
        # Axially aligned
        p_1 = [-kz[0], 0, kx]
        p_2 = [-kz[1], 0, kx]
        p_3 = [0, -kz[2], ky]
        p_4 = [0, -kz[3], ky]

    # p_1 = [-kz[0], 0, kx]
    # p_2 = [-kz[1], 0, kx]
    # p_3 = [0, -kz[2], ky]
    # p_4 = [0, -kz[3], ky]

    p = [p_1, p_2, p_3, p_4]
    p = [numpy.divide(pi, numpy.sqrt(numpy.dot(pi, pi))) for pi in p]

    assert numpy.isclose(numpy.dot(p[0], k[0]), 0)
    assert numpy.isclose(numpy.dot(p[1], k[1]), 0)
    assert numpy.isclose(numpy.dot(p[2], k[2]), 0)
    assert numpy.isclose(numpy.dot(p[3], k[3]), 0)

    return p
