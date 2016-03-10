__author__ = 'Pavel Dmitriev'

import numpy
import numpy.linalg
import PyATMM.isotropicTransferMatrix as isotropicTransferMatrix

def build_uniaxial_layer_matrix(e_o, e_e, w, kx, ky, d, opticAxis=([0., 1., 0.])):
    D = build_uniaxial_bounding_layer_matrix(e_o, e_e, w, kx, ky, opticAxis=opticAxis)
    P = build_uniaxial_propagation_matrix(e_o, e_e, w, kx, ky, d, opticAxis=opticAxis)

    #LayerMatrix = numpy.dot(numpy.linalg.inv(D_0), numpy.dot(numpy.dot(D, P), numpy.dot(numpy.linalg.inv(D), D_0)))

    LayerMatrix = numpy.dot(D, numpy.dot(P, numpy.linalg.inv(D)))
    return LayerMatrix


def build_uniaxial_bounding_layer_matrix(e_o, e_e, w, kx, ky, opticAxis=([0., 1., 0.])):
    D_0 = isotropicTransferMatrix.build_isotropic_dynamic_matrix(1, w, kx, ky)
    D = build_uniaxial_dynamic_matrix(e_o, e_e, w, kx, ky, opticAxis=opticAxis)

    return numpy.dot(numpy.linalg.inv(D_0), D)

#
# Building blocks
#
def build_uniaxial_propagation_matrix(e_o, e_e, w, kx, ky, d, opticAxis=([0., 1., 0.])):
    gamma = build_uniaxial_transmitted_wavevectors(e_o, e_e, w, kx, ky, opticAxis)

    P = numpy.asarray(
        [
            [numpy.exp(-1j*gamma[0]*d), 0, 0, 0],
            [0, numpy.exp(-1j*gamma[1]*d), 0, 0],
            [0, 0, numpy.exp(-1j*gamma[2]*d), 0],
            [0, 0, 0, numpy.exp(-1j*gamma[3]*d)]
        ], dtype=numpy.complex128
    )
    return P

def build_uniaxial_dynamic_matrix(e_o, e_e, w, kx, ky, opticAxis=([0., 1., 0.])):
    mu = 1.
    c = 299792458.  # m/c

    gamma = build_uniaxial_transmitted_wavevectors(e_o, e_e, w, kx, ky, opticAxis)
    k = [[kx, ky, g] for g in gamma]

    p = axially_aligned_uniaxial_polarizations(e_o, e_e, w, kx, ky, gamma, opticAxis)

    q = [(c/(w*mu))*numpy.cross(ki, pi) for ki, pi in zip(k, p)]

    D = numpy.asarray(
        [
            [p[0][0], p[1][0], p[2][0], p[3][0]],
            [q[0][1], q[1][1], q[2][1], q[3][1]],
            [p[0][1], p[1][1], p[2][1], p[3][1]],
            [q[0][0], q[1][0], q[2][0], q[3][0]]
        ], dtype=numpy.complex128
    )
    return D


def axially_aligned_uniaxial_polarizations(e_o, e_e, w, kx, ky, kz, opticAxis):

    # For now optic axis should be aligned to main axes
    #assert numpy.allclose(opticAxis, [0, 0, 1]) \
    #       or numpy.allclose(opticAxis, [0, 1, 0]) \
    #       or numpy.allclose(opticAxis, [1, 0, 0])
    # In general, as long as k-vector and optic axis are not colinear, this should work
    assert all(not numpy.allclose(opticAxis, [kx, ky, numpy.abs(g)]) for g in kz)
    assert numpy.isclose(numpy.dot(opticAxis, opticAxis), 1.)

    nu = (e_e - e_o) / e_o

    kap = [numpy.asarray([kx, ky, g]) for g in kz]
    kap = [numpy.divide(kap_i, numpy.sqrt(numpy.dot(kap_i, kap_i))) for kap_i in kap]
    ka = [numpy.dot(opticAxis, kap_i) for kap_i in kap]

    p_1 = numpy.cross(opticAxis, kap[0])
    p_2 = numpy.cross(opticAxis, kap[1])
    p_3 = numpy.subtract(opticAxis, ((1 + nu)/(1+nu*ka[2]**2))*ka[2] * kap[2])
    p_4 = numpy.subtract(opticAxis, ((1 + nu)/(1+nu*ka[3]**2))*ka[3] * kap[3])
    p = [p_1, p_2, p_3, p_4]

    p = [numpy.divide(pi, numpy.sqrt(numpy.dot(pi, pi))) for pi in p]
    return p


def build_uniaxial_transmitted_wavevectors(e_o, e_e, w, kx, ky, opticAxis=([0., 1., 0.])):
    mu = 1.
    c = 299792458.  # m/c

    nu = (e_e - e_o) / e_o
    k_par = numpy.sqrt(kx**2 + ky**2)
    k_0 = w/c
    n = [0, 0, 1]
    if not numpy.isclose(k_par, 0):
        l = [kx/k_par, ky/k_par, 0]
        assert numpy.isclose(numpy.dot(l, l), 1)
    else:
        l = [0, 0, 0]

    na = numpy.dot(n, opticAxis)
    la = numpy.dot(l, opticAxis)

    mod_kz_ord = numpy.sqrt(e_o * k_0**2 - k_par**2, dtype=numpy.complex128)

    mod_kz_extraord = (1 / (1 + nu * na**2)) * (-nu * k_par * na*la
                                                + numpy.sqrt(e_o * k_0**2 * (1 + nu) * (1 + nu * na**2)
                                                            - k_par**2 * (1 + nu * (la**2 + na**2)),
                                                            dtype=numpy.complex128)
                                               )


    k_z1 = mod_kz_ord
    k_z2 = -mod_kz_ord
    k_z3 = mod_kz_extraord
    k_z4 = -mod_kz_extraord
    return [k_z1, k_z2, k_z3, k_z4]
