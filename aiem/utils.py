import numpy as np
from scipy.optimize import fsolve


def s_from_eta(eta, r_le, s_offset=0):
    """ Transform eta coordinate to arc length along the fitted parabola with leading edge radius of r_le
    :param eta: eta coordinate in conformal space plane
    :param r_le: leading edge radius of the aerofoil
    :param s_offset: The offset between aerofoil curvilinear coordinate system and parabola arc length coordinate
                     system, present cambered aerofoils. Curvilinear coordinate of the parabola vertex.

    :return
    """
    s = 0.5 * r_le * (eta * np.sqrt(eta ** 2 + 1) + np.log(np.sqrt(eta ** 2 + 1) + eta)) + s_offset
    return s


def eta_from_s(s, r_le, s_offset=0):
    """

    :param s:
    :param r_le: leading edge radius
    :param s_offset: Curvilinear coordinate of the parabola vertex.
    :return:
    """
    eta = fsolve(_implicit_s, np.array([0]), args=(s, r_le, s_offset))
    return eta


def _implicit_s(eta, s0, r_le, s_offset):
    return s0-s_from_eta(eta, r_le, s_offset)

