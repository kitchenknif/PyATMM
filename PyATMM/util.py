import numpy


def solve_quartic(a, b, c, d, e):

    p = (8.*a*c - 3.*b**2) / (8.*a**2)
    q = (b**3 - 4.*a*b*c + 8.*a**2*d) / (8.*a**3)

    D0 = c**2 - 3*b*d + 12*a*e
    D1 = 2*c**3 - 9*b*c*d + 27*b**2*e + 27*a*d**2 - 72*a*d**2 - 72*a*c*e
    D = (1./27.)*(4*D0**3 - D1**2)

    if D != 0 and D0 == 0:
        Q = numpy.power((D1 + numpy.sign(D1)*numpy.sqrt(D1**2 - 4*D0**3))/2., 1./3.)
    else:
        Q = numpy.power((D1 + numpy.sqrt(D1**2 - 4*D0**3))/2., 1./3.)

    S = (1./2.) * numpy.sqrt(-(2./3.)*p + (1./(3.*a))*(Q + D0/Q))
    if S == 0:
        Q = -Q
        S = (1/2) * numpy.sqrt(-(2/3)*p + (1/(3*a))*(Q + D0/Q))

    x1 = -b/(4.*a) + S + (1./2.)*numpy.sqrt(-4.*S**2 - 2*p + q/S)
    x2 = -b/(4.*a) + S - (1./2.)*numpy.sqrt(-4.*S**2 - 2*p + q/S)
    x3 = -b/(4.*a) + S + (1./2.)*numpy.sqrt(-4.*S**2 - 2*p - q/S)
    x4 = -b/(4.*a) + S - (1./2.)*numpy.sqrt(-4.*S**2 - 2*p - q/S)

    return x1, x2, x3, x4