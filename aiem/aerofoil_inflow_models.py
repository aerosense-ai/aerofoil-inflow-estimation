import numpy as np
from scipy.optimize import least_squares


def estimate_stagnation_from_potential_flow(pressure, eta, apply_bounds=False):
    """Estimates the location of the stagnation point.
    
    :param eta: sensor locations in non-dimensional conformal plane coordinates (eta)
    :param pressure: pressure at sensor locations.
    :param i0: null sensor index in the pressure vector
    :param apply_bounds: Boolian. IF True, bound the estimation.
    :return: Stagnation point location in curvilinear coordinate
    """

    if not apply_bounds:
        bounds = (-np.inf, np.inf)
        sigma0 = 0
    else:
        # Some bounds to help the fitting:
        # sigma0 is initial guess
        if pressure[-1] < 0:
            sigma0 = -1
            bounds = (-2.5, 0.5)
        else:
            sigma0 = 1
            bounds = (-0.5, 2.5)

        # Central value must be the one at the leading edge!!!
        # TODO fix this bound for new algo
        # if abs(eta[np.where(pressure == np.amax(pressure))]) < 0.35:
        #     sigma0 = 0
        #     bounds = (-0.5, 0.5)


    popt = least_squares(_potential_flow_cost_function, sigma0,
                         args=(eta, pressure),
                         bounds=bounds,
                         verbose=1)

    return popt.x[0]


def estimate_speed_from_potential_flow(eta_stagnation, pressure, eta, rho):
    a = _potential_flow_function(eta_stagnation, eta)
    c = np.linalg.lstsq(np.atleast_2d(a).T, pressure, rcond=None)
    flow_speed = np.sqrt(2*abs(c[0])/rho)
    return flow_speed[0]


def _potential_flow_function(sigma_p, xdata):
    a = (xdata[1, :] - sigma_p) ** 2 / (1 + xdata[1, :] ** 2) - (xdata[0, :] - sigma_p) ** 2 / (1 + xdata[0, :] ** 2)
    return a


def _potential_flow_cost_function(sigma_p, xdata, ydata):
    """Cost function to perform non-linear least square optimisation
    c = 0.5*rho*U^2
    y_est = A*c

    :param sigma_p: location of the stagnation point. Parameter to optimise.
    :param xdata: pressure sensor locations in non-dimensional curvilinear coordinates (eta).
    :param ydata: pressure at sensor locations in Pa.
    :param i0: null sensor index in the pressure vector
    :return: cost to minimise
    """
    A = _potential_flow_function(sigma_p, xdata)
    c = np.linalg.lstsq(np.atleast_2d(A).T, ydata, rcond=None)  # Least square solution for linear parameters c
    y_est =A*c[0]

    return y_est - ydata
