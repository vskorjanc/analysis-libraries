
import numpy as np
import scipy.constants as phys
from bric_analysis_libraries import standard_functions as std

import numpy as np
from scipy.integrate import simpson


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


def calc_Jsc(df, am_df):
    '''
    Calculates Jsc by multiplying the EQE spectrum with AM1.5 spectrum and integrating. Returns values in mA cm-2.
    :param df: EQE spectra, indexed by energy (eV).
    :param am_df: Pandas dataframe which contains AM1.5 spectrum labeled as 'AM1.5G/(photons s-1 m-2 eV-1)'
    :returns: Pandas series which contains Jsc values.
    '''
    [jsc_df, am_ri] = std.common_reindex([df, am_df], fillna=np.nan)
    jsc_df = jsc_df.apply(
        lambda x: x * am_ri['AM1.5G/(photons s-1 m-2 eV-1)']).dropna(how='all')
    jsc_df = jsc_df.fillna(0)
    jsc = jsc_df.apply(lambda x: simpson(x, x.index)) * phys.e
    return jsc / 10
