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


def solve_transfer_matrix(M):
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

