__author__ = 'Pavel Dmitriev'

import numpy
import numpy.linalg

def build_anisotropic_permittivity(e1, e2, e3):
    eps = numpy.zeros([3, 3])
    eps[0, 0] = e1
    eps[1, 1] = e2
    eps[2, 2] = e3

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
    R = numpy.asarray(
        [numpy.cos(psi)*numpy.cos(phi) - numpy.cos(theta)*numpy.sin(phi)*numpy.sin(psi),
                -numpy.sin(psi)*numpy.cos(phi) - numpy.cos(theta)*numpy.sin(phi)*numpy.cos(psi),
                        numpy.sin(theta)*numpy.sin(phi)],
        [numpy.cos(psi)*numpy.sin(phi) + numpy.cos(theta)*numpy.cos(phi)*numpy.sin(psi),
                - numpy.sin(psi)*numpy.sin(phi) + numpy.cos(theta)*numpy.cos(phi)*numpy.cos(psi),
                        -numpy.sin(theta)*numpy.cos(phi)],
        [numpy.sin(theta)*numpy.sin(psi), numpy.sin(theta)*numpy.cos(psi), numpy.cos(theta)]
    )

    return R

def build_general_permittivity_matrix(e1, e2, e3, theta, phi, psi):
    E = build_anisotropic_permittivity(e1, e2, e3)
    R = build_rotation_matrix(theta, phi, psi)

    return numpy.dot(numpy.linalg.inv(R), numpy.dot(E, R))

