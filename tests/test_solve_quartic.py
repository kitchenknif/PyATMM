from unittest import TestCase
from util import *
__author__ = 'p.dmitriev'


class TestSolve_quartic(TestCase):
    def test_solve_quartic(self):
        a = 1
        b = 0
        c = 0
        d = 0
        e = -1
        print(solve_quartic(a, b, c, d, e))
        self.fail()
