
import numpy as np
import scipy.constants as phys
from bric_analysis_libraries import standard_functions as std


def calc_voc_rad(ratio):
    q = phys.e
    kb = phys.k
    return kb * 300 * np.log(ratio) / q


def calc_bb(x):
    '''
    Multiplies spectrum with black body photon flux.
    '''
    e = x.index
    pi = phys.pi
    h = phys.physical_constants['Planck constant in eV/Hz'][0]
    c = phys.c
    kb = phys.physical_constants['Boltzmann constant in eV/K'][0]
    return 2 * pi * e ** 2 / (h ** 3 * c ** 2 * (np.exp(e / (kb * 300)) - 1)) * x


def fit_lin(df, **kwargs):
    '''
    :param df: DataFrame with data for the fit.
    :param kwargs: Additional parameters to be passed to std.df_fit_function()
    and scipy.optimize.curve_fit().
    :returns: DataFrame with the fit parameter and error for each parameter.    
    '''
    def lin_reg(e, e_u, b):
        return e / e_u + b

    fit = std.df_fit_function(lin_reg, guess=(0.015, 1), **kwargs)
    return fit(df)


def fit_urbach_tail(data, fit_window, filter_window):
    '''Fits Urbach tail around the inflection point of EQE in log scale and
    calculates the '''

    log_data = data.apply(np.log)

    # find inflection point
    infl_point = log_data.rolling(
        window=filter_window,
        min_periods=int(filter_window/4),
        center=True
    ).mean().diff().idxmax()
    infl_point = float(infl_point)

    # fit within fit_window around the inflection point
    fit_data = log_data.loc[infl_point - fit_window:infl_point + fit_window]
    fit = fit_lin(fit_data)
    e_u = float(fit['e_u', 'value'])
    b = float(fit['b', 'value'])

    # calculate urbach tail for values below inflection point
    def calc_urbach_tail(data, e_u, b):
        data = data.copy()
        data.loc[:infl_point] = data.loc[:infl_point].apply(
            lambda x: np.exp(x.index / e_u + b))
        return data

    return (calc_urbach_tail(data, e_u, b), fit)
