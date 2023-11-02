import os
from unittest import TestCase


import numpy as np
import pandas as pd
from aiem import utils


class TestUtils(TestCase):
    def test_arc_length(self):
        s = utils.s_from_eta(-0.1, 0.18)
        self.assertAlmostEqual(s, -0.01803)

    def test_arc_length_with_offset(self):
        s = utils.s_from_eta(-0.1, 0.18, -1)
        self.assertAlmostEqual(s, -1.01803)

    def test_inverse_arc_length(self):
        eta = utils.eta_from_s(-0.01803, 0.18)
        self.assertAlmostEqual(eta, -0.1, delta=0.000001)

    def test_inverse_arc_length_with_offset(self):
        eta = utils.eta_from_s(-1.01803, 0.18, -1)
        self.assertAlmostEqual(eta, -0.1, delta=0.000001)
