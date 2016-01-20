__author__ = 'Pavel Dmitriev'

import numpy
import numpy.linalg
from transferMatrix import *

def isotropic_polarizations(w, eps, kx, ky, kz):

    #if not numpy.isclose(kx, 0):
    p_1 = [numpy.cos(kx/kz[0]), 0, numpy.abs(numpy.sin(kx/kz[0]))]
    p_2 = [numpy.cos(kx/kz[1]), 0, numpy.abs(numpy.sin(kx/kz[1]))]
    #else:
    #    p_1 = [1, 0, 0]
    #    p_2 = [1, 0, 0]
    #if not numpy.isclose(ky, 0):
    #    p_3 = [0, kz[2]/ky, 1]
    #    p_4 = [0, kz[3]/ky, 1]
    #else:
    #    p_3 = [0, 1, 0]
    #    p_4 = [0, 1, 0]
    p_3 = [0, numpy.cos(ky/kz[2]), numpy.abs(numpy.sin(ky/kz[2]))]
    p_4 = [0, numpy.cos(ky/kz[3]), numpy.abs(numpy.sin(ky/kz[3]))]
    p = [p_1, p_2, p_3, p_4]

    p = [numpy.divide(pi, numpy.sqrt(numpy.dot(pi, pi))) for pi in p]
    return p



def build_isotropic_layer_matrix(eps, w, kx, ky, d):
    mu = 1.
    c = 299792458.  # m/c

    mod_kz = numpy.sqrt(eps*(w/c)**2 - kx**2 - ky**2)
    gamma = [mod_kz, -mod_kz, mod_kz, -mod_kz]
    k = [numpy.asarray([kx, ky, g]) for g in gamma]

    #
    # Build polarization vectors
    #

    p = isotropic_polarizations(w, eps, kx, ky, gamma)
    #p = [build_polarization_vector(w, build_general_permittivity_matrix(eps, eps, eps, 0, 0, 0), kx, ky, g, mu) for g in gamma]
    q = [(c/(w*mu))*numpy.cross(ki, pi) for ki, pi in zip(k, p)]

    #
    # Build boundary transition matrix
    #
    D_0 = build_vacuum_matrix(w, kx, ky)

    D = numpy.asarray(
        [
            [p[0][0], p[1][0], p[2][0], p[3][0]],
            [q[0][1], q[1][1], q[2][1], q[3][1]],
            [p[0][1], p[1][1], p[2][1], p[3][1]],
            [q[0][0], q[1][0], q[2][0], q[3][0]]
        ], dtype=numpy.complex128
    )
    #
    # Build propagation matrix
    #
    P = numpy.asarray(
        [
            [numpy.exp(-1j*gamma[0]*d), 0, 0, 0],
            [0, numpy.exp(-1j*gamma[1]*d), 0, 0],
            [0, 0, numpy.exp(-1j*gamma[2]*d), 0],
            [0, 0, 0, numpy.exp(-1j*gamma[3]*d)]
        ], dtype=numpy.complex128
    )

    #
    # Multiply matricies
    #
    LayerMatrix = numpy.linalg.inv(D_0)
    LayerMatrix = numpy.dot(LayerMatrix, numpy.dot(numpy.dot(D, P), numpy.dot(numpy.linalg.inv(D), D_0)))
    return LayerMatrix

def build_isotropic_bounding_layer_matrix(eps, w, kx, ky):
    mu = 1.
    c = 299792458.  # m/c

    mod_kz = numpy.sqrt(eps*(w/c)**2 - kx**2 - ky**2)
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
    D_0 = build_vacuum_matrix(w, kx, ky)

    D = numpy.asarray(
        [
            [p[0][0], p[1][0], p[2][0], p[3][0]],
            [q[0][1], q[1][1], q[2][1], q[3][1]],
            [p[0][1], p[1][1], p[2][1], p[3][1]],
            [q[0][0], q[1][0], q[2][0], q[3][0]]
        ], dtype=numpy.complex128
    )
    return numpy.dot(numpy.linalg.inv(D_0), D)

def build_isotropic_propagation_matrix(eps, w, kx, ky, d):
    mu = 1.
    c = 299792458.  # m/c

    mod_kz = numpy.sqrt(eps*(w/c)**2 - kx**2 - ky**2)
    gamma = [mod_kz, -mod_kz, mod_kz, -mod_kz]
    k = [numpy.asarray([kx, ky, g]) for g in gamma]

    #
    # Build propagation matrix
    #
    P = numpy.asarray(
        [
            [numpy.exp(-1j*gamma[0]*d), 0, 0, 0],
            [0, numpy.exp(-1j*gamma[1]*d), 0, 0],
            [0, 0, numpy.exp(-1j*gamma[2]*d), 0],
            [0, 0, 0, numpy.exp(-1j*gamma[3]*d)]
        ], dtype=numpy.complex128
    )

    return P

def build_vacuum_matrix(w, kx, ky):
    mu = 1.
    c = 299792458.  # m/c

    mod_kz = numpy.sqrt((w/c)**2 - kx**2 - ky**2)
    gamma = [mod_kz, -mod_kz, mod_kz, -mod_kz]
    k = [numpy.asarray([kx, ky, g]) for g in gamma]

    #
    # Build polarization vectors
    #

    p = isotropic_polarizations(w, 1, kx, ky, gamma)
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