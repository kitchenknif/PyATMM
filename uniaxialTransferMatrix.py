__author__ = 'Pavel Dmitriev'

import numpy
import numpy.linalg
import isotropicTransferMatrix

def axially_aligned_uniaxial_polarizations(w, e_o, e_e, kx, ky, kz, opticAxis):

    # For now optic axis should be aligned to main axes
    #assert numpy.allclose(opticAxis, [0, 0, 1]) \
    #       or numpy.allclose(opticAxis, [0, 1, 0]) \
    #       or numpy.allclose(opticAxis, [1, 0, 0])
    # In general, as long as k-vector and optic axis are not colinear, this should work
    assert all(not numpy.allclose(opticAxis, [kx, ky, numpy.abs(g)]) for g in kz)

    kap = [numpy.asarray([kx, ky, g]) for g in kz]
    kap = [numpy.divide(kap_i, numpy.sqrt(numpy.dot(kap_i, kap_i))) for kap_i in kap ]

    p_1 = numpy.cross(opticAxis, kap[0])
    p_2 = numpy.cross(opticAxis, kap[1])
    p_3 = numpy.subtract(opticAxis, (e_e/e_o)*numpy.dot(opticAxis, kap[2]) * kap[2])
    p_4 = numpy.subtract(opticAxis, (e_e/e_o)*numpy.dot(opticAxis, kap[3]) * kap[3])
    p = [p_1, p_2, p_3, p_4]

    p = [numpy.divide(pi, numpy.sqrt(numpy.dot(pi, pi))) for pi in p]
    return p


def build_uniaxial_transmitted_wavevectors(e_o, e_e, w, kx, ky, opticAxis=([0., 1., 0.])):
    mu = 1.
    c = 299792458.  # m/c

    nu = (e_e - e_o)/e_o
    k_par = numpy.sqrt(kx**2 + ky**2)
    k_0 = w/c
    opt_par = numpy.sqrt(opticAxis[0]**2 + opticAxis[1]**2)
    opt_per = numpy.dot(opticAxis, [0, 0, 1])

    mod_kz_ord = numpy.sqrt(e_o*(w/c)**2 - kx**2 - ky**2)
    mod_kz_extraord = (1/(1 + nu*opt_per**2))*(-nu*opt_per*opt_par
                                                + numpy.sqrt(e_o*k_0**2*(1+nu)*(1+nu*opt_per**2)
                                                    -k_par**2*(1+nu*(opt_per**2 + opt_par**2))))

    k_z1 = mod_kz_ord
    k_z2 = -mod_kz_ord
    k_z3 = mod_kz_extraord
    k_z4 = -mod_kz_extraord
    return [k_z1, k_z2, k_z3, k_z4]


def build_uniaxial_layer_matrix(e_o, e_e, w, kx, ky, d, opticAxis=([0., 1., 0.])):
    mu = 1.
    c = 299792458.  # m/c

    gamma = build_uniaxial_transmitted_wavevectors(e_o, e_e, w, kx, ky, opticAxis)
    k = [[kx, ky, g] for g in gamma]

    p = axially_aligned_uniaxial_polarizations(w, e_o, e_e, kx, ky, gamma, opticAxis)

    q = [(c/(w*mu))*numpy.cross(ki, pi) for ki, pi in zip(k, p)]

    D_0 = isotropicTransferMatrix.build_vacuum_matrix(w, kx, ky)
    D = numpy.asarray(
        [
            [p[0][0], p[1][0], p[2][0], p[3][0]],
            [q[0][1], q[1][1], q[2][1], q[3][1]],
            [p[0][1], p[1][1], p[2][1], p[3][1]],
            [q[0][0], q[1][0], q[2][0], q[3][0]]
        ], dtype=numpy.complex128
    )
    # print("D:", numpy.real(D))

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
    #
    # Multiply matricies
    #
    LayerMatrix = numpy.linalg.inv(D_0)
    LayerMatrix = numpy.dot(LayerMatrix, numpy.dot(numpy.dot(D, P), numpy.dot(numpy.linalg.inv(D), D_0)))
    #return LayerMatrix
    return numpy.dot(numpy.linalg.inv(D_0), D)