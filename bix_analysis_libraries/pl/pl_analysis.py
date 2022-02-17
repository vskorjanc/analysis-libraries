import scipy.constants as phys
import numpy as np
from plotly import express as px

from bric_analysis_libraries import standard_functions as std
from .. import bix_standard_functions as bsf


def lin_reg(e, t, qfls):
    kb = phys.physical_constants['Boltzmann constant in eV/K'][0]
    return - e / (kb * t) + qfls / (kb * t)


def fit_lin(df, **kwargs):
    '''
    :param df: DataFrame with data for the fit.
    :param kwargs: Additional parameters to be passed to std.df_fit_function()
    and scipy.optimize.curve_fit().
    :returns: DataFrame with the fit parameter and error for each parameter.    
    '''
    fit = std.df_fit_function(lin_reg, **kwargs)
    return fit(df)


def calc_log(data):
    def calc(x):
        h = phys.physical_constants['Planck constant in eV/Hz'][0]
        const = (h ** 3) * (phys.c ** 2) / (2 * phys.pi)
        return np.log(x * const / (x.index.values ** 2))

    data = data.apply(calc)
    return data.replace([-np.inf, np.inf], np.nan).dropna(how='all')


def high_energy_tail_fit(data, window, start=None, end=None, temperature=300, temp_window=1):
    '''
    Calculates ln(flux * constant) and applies linear fit on a window around 
    the inflection point to determine QFLS.  
    :param data: DataFrame or Series with flux of a single measurement.
    :param window: Window in eV around the inflection point to take into account for fit. 
    :param start: Lower bound of the search range, or None. [Default: None]
    :param end: Upper bound of the search range, or None. [Default: None]
    :param temperature: Temperature in Kelvin. [Default: 300]
    :param temp_window: Window for determining temperature bounds when. [Default: 1]
    :returns: DataFrame with parameters qfls and t and error for each parameter.
    '''
    data = calc_log(data)

    # def find inflection as separate?
    svg_data = bsf.apply_savgol(data, deriv=1)
    infl_point = svg_data.loc[start:end].idxmin().values[0]

    fit_data = data.loc[infl_point - window: infl_point + window]
    guess = (temperature, 1)
    bounds = [[temperature - temp_window, 0], [temperature + temp_window, 1.5]]
    return fit_lin(fit_data, guess=guess, bounds=bounds)


def plot_hetf(data, fit, start=None, end=None):
    '''
    Plots log of flux and linear fit for visual inspection.
    :param data: DataFrame with flux of a single measurement.
    :param fit: DataFrame given by `high_energy_tail_fit` function.
    :param start: Lower bound of the search range, or None. [Default: None]
    :param end: Upper bound of the search range, or None. [Default: None]
    :returns: Plotly figure.
    '''
    pl_data = calc_log(data.loc[start: end])
    pl_data.columns = ['log']
    pl_data['fit'] = lin_reg(pl_data.index, fit.t.value[0], fit.qfls.value[0])
    fig = px.line(pl_data)
    return fig


def calculate_plqy(df, ref, emission_range, excitation_range):
    '''
    Calculates PLQY.
    :param df: DataFrame with measured PL spectra.
    :param ref: DataFrame with reference spectrum.
    :param emission_range: Tuple with start and end values for 
    integrating emission.
    :param excitation_range: Tuple with start and end values for 
    integrating excitation.
    :returns: Pandas Series with calculated PLQY
    '''

    emitted = bsf.simpson_integrate(df, *emission_range)

    excitation_before = bsf.simpson_integrate(ref, *excitation_range)
    excitation_before = float(excitation_before)

    excitation_after = bsf.simpson_integrate(df, *excitation_range)

    return emitted / (excitation_before - excitation_after)
