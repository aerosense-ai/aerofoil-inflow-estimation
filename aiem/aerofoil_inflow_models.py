import numpy as np
from scipy.optimize import least_squares


def estimate_from_potential_flow(pressure, eta, i0):
    """Estimates the location of the stagnation point.
    
    :param eta: sensor locations in non-dimensional curvilinear coordinates (eta)
    :param pressure: pressure at sensor locations.
    :param i0: null sensor index in the pressure vector
    :return: Stagnation point location in curvilinear coordinate
    """

    # Some bounds to help the fitting:
    if pressure[-1] < 0:
        sigma0 = -1
        bounds = (-2.5, 0.5)
    else:
        sigma0 = 1
        bounds = (0.5, 2.5)

    # Central value must be the one at the leading edge!!!
    if abs(eta[np.where(pressure == np.amax(pressure))]) < 0.35:
        sigma0 = 0
        bounds = (-0.5, 0.5)

    popt = least_squares(potential_flow_cost_function, sigma0,
                         args=(eta, pressure, i0),
                         bounds=bounds,
                         verbose=1)

    return popt.x[0]


def potential_flow_cost_function(sigma_p, xdata, ydata, i0):
    """Cost function to perform non-linear least square optimisation
    c = 0.5*rho*U^2
    y_est = A*c

    :param sigma_p: location of the stagnation point. Parameter to optimise.
    :param xdata: pressure sensor locations in non-dimensional curvilinear coordinates (eta).
    :param ydata: pressure at sensor locations in Pa.
    :param i0: null sensor index in the pressure vector
    :return: cost to minimise
    """
    A = (xdata[i0] - sigma_p) ** 2 / (1 + xdata[i0] ** 2) - (xdata - sigma_p) ** 2 / (1 + xdata ** 2)
    c = np.linalg.lstsq(np.atleast_2d(A).T, ydata)  # Least square solution for linear parameters c
    y_est =A*c[0]

    return y_est - ydata
