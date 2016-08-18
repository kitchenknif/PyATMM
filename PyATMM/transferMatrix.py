__author__ = 'Pavel Dmitriev'

import numpy
import numpy.linalg

def build_anisotropic_permittivity(e1, e2, e3):
    eps = numpy.zeros([3, 3], dtype=numpy.complex128)
    eps[0, 0] = e1
    eps[1, 1] = e2
    eps[2, 2] = e3
    return eps

def build_rotation_matrix(theta, phi, psi):
    # Rx = numpy.asarray(
    #     [1, 0, 0],
    #     [0, numpy.cos(theta), -numpy.sin(theta)],
    #     [0, numpy.sin(theta), numpy.cos(theta)]
    # )
    #
    # Ry = numpy.asarray(
    #     [numpy.cos(phi), 0, numpy.sin(phi)],
    #     [0, 1, 0],
    #     [-numpy.sin(phi), numpy.cos(phi)]
    # )
    #
    # Rz = numpy.asarray(
    #     [numpy.cos(psi), -numpy.sin(psi), 0],
    #     [numpy.sin(psi), numpy.cos(psi), 0],
    #     [0, 0, 1]
    # )
    #
    # R = numpy.dot(Rz, numpy.dot(Ry, Rx))
    R = numpy.asarray([
        [numpy.cos(psi)*numpy.cos(phi) - numpy.cos(theta)*numpy.sin(phi)*numpy.sin(psi),
                -numpy.sin(psi)*numpy.cos(phi) - numpy.cos(theta)*numpy.sin(phi)*numpy.cos(psi),
                        numpy.sin(theta)*numpy.sin(phi)],
        [numpy.cos(psi)*numpy.sin(phi) + numpy.cos(theta)*numpy.cos(phi)*numpy.sin(psi),
                - numpy.sin(psi)*numpy.sin(phi) + numpy.cos(theta)*numpy.cos(phi)*numpy.cos(psi),
                        -numpy.sin(theta)*numpy.cos(phi)],
        [numpy.sin(theta)*numpy.sin(psi), numpy.sin(theta)*numpy.cos(psi), numpy.cos(theta)]
    ], dtype=numpy.float64)

    return R

def build_general_permittivity_matrix(e1, e2, e3, theta, phi, psi):
    E = build_anisotropic_permittivity(e1, e2, e3)
    R = build_rotation_matrix(theta, phi, psi)

    Eps = numpy.dot(R, numpy.dot(E, numpy.linalg.inv(R)))

    assert numpy.isclose(Eps[1, 0], Eps[0, 1])
    assert numpy.isclose(Eps[2, 0], Eps[0, 2])
    assert numpy.isclose(Eps[1, 2], Eps[2, 1])

    return Eps

def build_polarization_vector(w, Eps, kx, ky, g, mu):

    e_zz = Eps[2, 2]
    e_yy = Eps[1, 1]
    e_xx = Eps[0, 0]
    e_xy = Eps[0, 1]
    e_yz = Eps[1, 2]
    e_xz = Eps[0, 2]

    # TODO: Guaranteed bullshit?
    p = numpy.asarray(
        [(w**2*e_yy - kx**2 - g**2)*(w**2*e_zz - kx**2 - ky**2) - (w**2*e_yz + ky*g)**2,
         (w**2*e_yz + ky*g)*(w**2*e_xz + kx*g) - (w**2*e_xy + kx*ky)*(w**2*e_zz - kx**2 - ky**2),
         (w**2*e_xy + kx*ky)*(w**2*e_yz + ky*g) - (w**2*e_xz + kx*g)*(w**2*e_yy - kx**2 - g**2)],
        dtype=numpy.complex128)
    p = numpy.divide(p, numpy.sqrt(numpy.dot(p, p)))
    return p

def build_anisotropic_layer_matrix(e1, e2, e3, theta, phi, psi, w, kx, ky, d):
    #
    # Build epsilon matrix
    #
    Eps = build_general_permittivity_matrix(e1, e2, e3, theta, phi, psi)

    e_zz = Eps[2, 2]
    e_yy = Eps[1, 1]
    e_xx = Eps[0, 0]
    e_xy = Eps[0, 1]
    e_yz = Eps[1, 2]
    e_xz = Eps[0, 2]
    # should be symmetric, so we don't need e_zy, e_zx, e_yx

    #
    # Solve for eigenmodes
    #   Solve quartic equation A*kz**4 + B*kz**3 + C*kz**2 + D*kz + E
    #   Coefficients from Determinant of matrix
    #   [(w/c)**2 * E + inner(k, k) * I + outer(k, k)]
    a = w**2 * e_zz                        # *kz**4

    b = w**2 * (2*e_xz*kx + 2*e_yz*ky)     # *kz**3

    c = w**2 * ( -e_xx*e_zz*w**2 + e_xx*kx**2 + 2*e_xy*kx*ky + e_xz**2*w**2 - e_yy*e_zz*w**2 + e_yy*ky**2
                    + e_yz**2*w**2 + e_zz*kx**2+e_zz*ky**2
                )                          # *kz**2

    d = w**2 * ( -2*e_xx*e_yz*w**2*ky + 2*e_xy*e_xz*w**2*ky + 2*e_xy*e_yz**w**2*kx - 2*e_xz*e_yy*w**2*kx + 2*e_xz*kx**3
                    + 2*e_xz*kx*ky**2 + 2*e_yz*kx**2*ky + 2*e_yz*ky**3
                )                          # *kz**1

    e = w**2 * (e_xx*e_yy*e_zz*w**4 - e_xx*e_yy*w**2*kx**2 - e_xx*e_yy*w**2*ky**2 - e_xx*e_yz**2*w**4
                    - e_xx*e_zz*w**2*kx**2 + e_xx*kx**4 + e_xx*kx**2*ky**2 - e_xy**2*e_zz*w**4 + e_xy**2*w**2*kx**2
                    + e_xy**2*w**2*ky**2 + 2*e_xy*e_xz*e_yz*w**4
                    - 2*e_xy*e_zz*w**2*kx*ky + 2*e_xy*kx**3*ky + 2*e_xy*kx*ky**3 - e_xz**2*e_yy*w**4
                    + e_xz**2*w**2*kx**2 + 2*e_xz*e_yz*w**2*kx*ky - e_yy*e_zz*w**2*ky**2 + e_yy*kx**2*ky**2
                    + e_yy*ky**4 + e_yz**2*w**2*ky**2
                )                           # *kz**0
    coeffs = [a, b, c, d, e]
    gamma = numpy.roots(coeffs)

    #TODO: Hack needs fixing
    tmp = gamma[2]
    gamma[2] = gamma[1]
    gamma[1] = tmp

    #
    # Build polarization vectors
    #
    mu = 1
    c = 1 # m/c

    p = [build_polarization_vector(w, Eps, kx, ky, g, mu) for g in gamma]
    print("P:", numpy.real(p))
    q = [(c/(w*mu))*numpy.cross([kx, ky, gi], pi) for gi, pi in zip(gamma, p)]
    print("Q:", numpy.real(q))

    #return p, gamma, Eps

    #
    # Tests
    #
    #vec = numpy.dot(Eps, p[0])
    #print( numpy.dot(vec, [kx, ky, gamma[0]]) )
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
    print("D:", numpy.real(D))

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


def solve_transfer_matrix(M):

    denom = M[0, 0]*M[2, 2] - M[0, 2]*M[2, 0]
    #print(denom)


    #
    # Components 3,4 - S
    # Components 1,2 - P Was mixed up until 11.08.2016 (switched s and p)
    #
    r_pp = (M[1, 0]*M[2, 2] - M[1, 2]*M[2, 0]) / (M[0, 0]*M[2, 2] - M[0, 2]*M[2, 0])
    r_ps = (M[3, 0]*M[2, 2] - M[3, 2]*M[2, 0]) / (M[0, 0]*M[2, 2] - M[0, 2]*M[2, 0])
    r_sp = (M[0, 0]*M[1, 2] - M[1, 0]*M[0, 2]) / (M[0, 0]*M[2, 2] - M[0, 2]*M[2, 0])
    r_ss = (M[0, 0]*M[3, 2] - M[3, 0]*M[0, 2]) / (M[0, 0]*M[2, 2] - M[0, 2]*M[2, 0])
    t_pp = M[2, 2] / (M[0, 0]*M[2, 2] - M[0, 2]*M[2, 0])
    t_ps = -M[2, 0] / (M[0, 0]*M[2, 2] - M[0, 2]*M[2, 0])
    t_sp = -M[0, 2] / (M[0, 0]*M[2, 2] - M[0, 2]*M[2, 0])
    t_ss = M[0, 0] / (M[0, 0]*M[2, 2] - M[0, 2]*M[2, 0])

    return r_ss, r_sp, r_ps, r_pp, t_ss, t_sp, t_ps, t_pp

