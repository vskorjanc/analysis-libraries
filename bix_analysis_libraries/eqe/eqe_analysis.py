import numpy as np
import scipy.constants as phys
from scipy.optimize import curve_fit
from bric_analysis_libraries import standard_functions as std
from bix_analysis_libraries import bix_standard_functions as bsf

import pandas as pd
import numpy as np
from scipy.integrate import simpson


def calc_voc_rad(ratio):
    q = phys.e
    kb = phys.k
    return kb * 300 * np.log(ratio) / q


def calc_bb(x):
    """
    Multiplies spectrum with black body photon flux.
    """
    e = x.index
    pi = phys.pi
    h = phys.physical_constants["Planck constant in eV/Hz"][0]
    c = phys.c
    kb = phys.physical_constants["Boltzmann constant in eV/K"][0]
    return 2 * pi * e**2 / (h**3 * c**2 * (np.exp(e / (kb * 300)) - 1)) * x


def fit_lin(df, **kwargs):
    """
    :param df: DataFrame with data for the fit.
    :param kwargs: Additional parameters to be passed to std.df_fit_function()
    and scipy.optimize.curve_fit().
    :returns: DataFrame with the fit parameter and error for each parameter.
    """

    def lin_reg(e, e_u, b):
        return e / e_u + b

    fit = std.df_fit_function(lin_reg, guess=(0.015, 1), **kwargs)
    return fit(df)


def fit_urbach_tail(data, fit_window, filter_window):
    """Fits Urbach tail around the inflection point of EQE in log scale and
    calculates the"""

    log_data = data.apply(np.log)

    # find inflection point
    infl_point = (
        log_data.rolling(
            window=filter_window, min_periods=int(filter_window / 4), center=True
        )
        .mean()
        .diff()
        .idxmax()
    )
    infl_point = float(infl_point.iloc[0])

    # fit within fit_window around the inflection point
    fit_data = log_data.loc[infl_point - fit_window : infl_point + fit_window]
    fit = fit_lin(fit_data)
    e_u = float(fit["e_u", "value"].iloc[0])
    b = float(fit["b", "value"].iloc[0])

    # calculate urbach tail for values below inflection point
    def calc_urbach_tail(data, e_u, b):
        data = data.copy()
        data.loc[:infl_point] = data.loc[:infl_point].apply(
            lambda x: np.exp(x.index / e_u + b)
        )
        return data

    return (calc_urbach_tail(data, e_u, b), fit)


def calc_Jsc(df, am_df):
    """
    Calculates Jsc by multiplying the EQE spectrum with AM1.5 spectrum and integrating. Returns values in mA cm-2.
    :param df: EQE spectra, indexed by energy (eV).
    :param am_df: Pandas dataframe which contains AM1.5 spectrum labeled as 'AM1.5G/(photons s-1 m-2 eV-1)'
    :returns: Pandas series which contains Jsc values.
    """
    [jsc_df, am_ri] = std.common_reindex([df, am_df], fillna=np.nan)
    jsc_df = jsc_df.apply(lambda x: x * am_ri["AM1.5G/(photons s-1 m-2 eV-1)"]).dropna(
        how="all"
    )
    jsc_df = jsc_df.fillna(0)
    jsc = jsc_df.apply(lambda x: simpson(x, x.index)) * phys.e
    return jsc / 10


def sigmoid_function(x, amplitude, midpoint, steepness):
    return amplitude / (1 + np.exp(-2.63 * (x - midpoint) / steepness))


def fit_sigmoid(series, area_width):
    series = series.dropna()
    interpol_series = bsf.interpolate(series, 0.001, "cubic")
    # take inflection point as the midpoint guess
    midpoint_guess = (
        bsf.apply_savgol(interpol_series, window_length=200, deriv=1).idxmax().values[0]
    )

    amplitude_guess = series.max()
    if amplitude_guess > 1:
        amplitude_guess = 1
    steepness_guess = 0.04

    selected = series.loc[midpoint_guess - area_width : midpoint_guess + area_width]

    x_data = selected.index.values
    y_data = selected.values

    p0 = [amplitude_guess, midpoint_guess, steepness_guess]

    popt, _ = curve_fit(
        sigmoid_function, x_data, y_data, p0=p0, bounds=([0, 1.5, 0], [1, 2.2, 0.1])
    )
    return popt, midpoint_guess


def apply_sigmoid_fit(df):
    fit = pd.DataFrame(
        index=pd.Index(["amplitude", "midpoint", "steepness"]), columns=df.columns
    )
    sigmoid_tail_df = df.copy()
    sigmoid_df = pd.DataFrame(
        index=np.arange(df.index.min(), df.index.max(), 0.001), columns=df.columns
    )
    area_width = 0.1
    for column in df.columns:
        popt, midpoint_guess = fit_sigmoid(df[column], area_width)
        fit[column] = popt
        # set the values below the inflection point to the fit values for J0 calculation
        sigmoid_tail_df.loc[: popt[1], column] = [
            sigmoid_function(x, *popt)
            for x in sigmoid_tail_df.loc[: popt[1]].index.values
        ]
        sigmoid_df.loc[
            midpoint_guess - area_width : midpoint_guess + area_width, column
        ] = [
            sigmoid_function(x, *popt)
            for x in sigmoid_df.loc[
                midpoint_guess - area_width : midpoint_guess + area_width
            ].index.values
        ]

    sigmoid_tail_df = bsf.interpolate(sigmoid_tail_df, 0.001, "cubic")
    metrics = pd.DataFrame(fit.T[["midpoint", "steepness"]])
    metrics = metrics.rename({"midpoint": "bandgap/eV"}, axis=1)
    return sigmoid_df, sigmoid_tail_df, metrics
