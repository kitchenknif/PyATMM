__author__ = 'Pavel Dmitriev'

import numpy
import numpy.linalg

def axially_aligned_uniaxial_polarizations(w, e_o, e_e, kx, ky, kz, opticAxis):

    # For now optic axis should be aligned to main axes
    assert numpy.allclose(opticAxis, [0, 0, 1]) \
           or numpy.allclose(opticAxis, [0, 1, 0]) \
           or numpy.allclose(opticAxis, [1, 0, 0])
    # In general, as long as k-vector and optic axis are not colinear, this should work
    assert all(not numpy.allclose(opticAxis, [kx, ky, numpy.abs(g)]) for g in kz)

    kap = [ numpy.asarray([kx, ky, g]) for g in kz ]
    kap = [numpy.divide(kap_i, numpy.sqrt(numpy.dot(kap_i, kap_i))) for kap_i in kap ]

    p_1 = numpy.cross(opticAxis, kap[0])
    p_2 = numpy.cross(opticAxis, kap[1])
    p_3 = numpy.subtract(opticAxis, (e_e/e_o)*numpy.dot(opticAxis, kap[2]) * kap[2])
    p_4 = numpy.subtract(opticAxis, (e_e/e_o)*numpy.dot(opticAxis, kap[3]) * kap[3])
    p = [p_1, p_2, p_3, p_4]

    p = [numpy.divide(pi, numpy.sqrt(numpy.dot(pi, pi))) for pi in p]
    return p


def solve_axially_aligned_eigenmodes(e_1, e_2, e_3, w, kx, ky):
    kz_4 = (e_3)

    kz_2 = (-e_1*e_3*w**2 + e_1*kx*2 - e_2*e_3*w**2 + e_2*ky**2 + e_3*kx**2 + e_3*ky**2)

    kz_0 = (e_1*e_2*e_3*w**4 - e_1*e_2*w**2*kx**2 - e_1*e_2*w**2*ky**2 - e_1*e_3*w**2*kx**2 + e_1*kx**4
                 + e_1*kx**2*ky**2 - e_2*e_3*w**2*ky**2 + e_2*kx**2*ky**2 + e_2*ky**4)

    # biquadratic equation
    kz2_1 = (-kz_2 + numpy.sqrt(kz_2**2 - 4*kz_4*kz_0))/2.
    kz2_2 = (-kz_2 - numpy.sqrt(kz_2**2 - 4*kz_4*kz_0))/2.

    return [numpy.sqrt(kz2_1), -numpy.sqrt(kz2_1), numpy.sqrt(kz2_2), -numpy.sqrt(kz2_2)]


def build_uniaxial_layer_matrix(e_o, e_e, w, kx, ky, d, opticAxis=([0., 1., 0.])):
    mu = 1.
    c = 299792458.  # m/c

    #
    # Solve for eigenmodes
    #   Solve quartic equation A*kz**4 + B*kz**3 + C*kz**2 + D*kz + E
    #   Coefficients from Determinant of matrix
    #   [(w/c)**2 * E + inner(k, k) * I + outer(k, k)]

    # For now optic axis should be aligned to main axes
    assert numpy.allclose(opticAxis, [0, 0, 1]) \
           or numpy.allclose(opticAxis, [0, 1, 0]) \
           or numpy.allclose(opticAxis, [1, 0, 0])
    # In general, as long as k-vector and optic axis are not colinear, this should work
    if opticAxis[0] != 0:
        gamma = solve_axially_aligned_eigenmodes(e_e, e_o, e_o, w, kx, ky)
    elif opticAxis[1] != 0:
        gamma = solve_axially_aligned_eigenmodes(e_o, e_e, e_o, w, kx, ky)
    elif opticAxis[2] != 0:
        gamma = solve_axially_aligned_eigenmodes(e_o, e_o, e_e, w, kx, ky)
    k = [numpy.asarray([kx, ky, g]) for g in gamma]

    #[print("kz:", kz) for kz in gamma]

    #
    # Build polarization vectors
    #

    p = axially_aligned_uniaxial_polarizations(w, e_o, e_e, kx, ky, gamma, opticAxis)
    #[print("P:", pi) for pi in p]

    #print([(ki, pi) for ki, pi in zip(k, p)])

    q = [(c/(w*mu))*numpy.cross(ki, pi) for ki, pi in zip(k, p)]
    #[print("Q:", qi) for qi in q]

    # return p, gamma, Eps

    #
    # Tests
    #
    # vec = numpy.dot(Eps, p[0])
    # print( numpy.dot(vec, [kx, ky, gamma[0]]) )
    #
    #
    #

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
    LayerMatrix = numpy.dot(D, numpy.dot(P, numpy.linalg.inv(D)))
    #return LayerMatrix
    return D