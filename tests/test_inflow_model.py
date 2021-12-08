import os
from unittest import TestCase


import numpy as np
import pandas as pd
from aiem.aerofoil_inflow_models import estimate_from_potential_flow


class TestModel(TestCase):
    def test_estimate_from_pf(self):
        """Test that the algorythm produces expected result on a benchmark NACA 0018 case"""

        # Some data to test the script:
        x_i = np.array([0.0376, 0.0065, 0.0, 0.0132, 0.0563])
        y_i = np.array([-0.0471j, -0.0207j, 0.0j, 0.0291j, 0.0561j])
        dpressures_pos = x_i + y_i

        th = 0.18  # NACA 0018
        r_le = 1.1019 * th ** 2

        # Transform x,y into curvilinear (eta)
        eta = np.sqrt(2 * np.real(dpressures_pos) / r_le) * np.sign(np.imag(dpressures_pos))  # using x formula

        mu_i = np.array([0, 300.44, -37.10, -1024.5, -1108.63])

        eta_stagnation = estimate_from_potential_flow(mu_i, eta, 0)

        self.assertAlmostEqual(eta_stagnation, -0.553, delta=0.001)

    def test_estimate_from_pf_with_pandas(self):
        """Test that estimate function can be passed as a pandas argument"""
        # Some data to test the script:
        x_i = np.array([0.0376, 0.0065, 0.0, 0.0132, 0.0563])
        y_i = np.array([-0.0471j, -0.0207j, 0.0j, 0.0291j, 0.0561j])
        dpressures_pos = x_i + y_i

        th = 0.18  # NACA 0018
        r_le = 1.1019 * th ** 2

        # Transform x,y into curvilinear (eta)
        eta = np.sqrt(2 * np.real(dpressures_pos) / r_le) * np.sign(np.imag(dpressures_pos))  # using x formula

        measurements_df = pd.DataFrame(
            {"p_sensor_0": [0, 0, 0, 0],
             "p_sensor_1": [300.44, 300.44, 300.44, 300.44],
             "p_sensor_2": [-37.10, -37.10, -37.10, -37.10],
             "p_sensor_3": [-1024.5, -1024.5, -1024.5, -1024.5],
             "p_sensor_4": [-1108.63, -1108.63, -1108.63, -1108.63]}
        )

        measurements_df['eta_stagnation'] = measurements_df.apply(estimate_from_potential_flow, args=(eta, 0), axis=1)

        self.assertIsNone(np.testing.assert_array_almost_equal(np.array([-0.553, -0.553, -0.553, -0.553]),
                                                             measurements_df['eta_stagnation'],
                                                             decimal=2))







