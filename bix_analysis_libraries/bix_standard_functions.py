import math
import pandas as pd
import numpy as np


# Pandas functions


def add_level (df, value, name, axis = 0):
    '''
    Adds level with a single value to the index of a DataFrame.
    :param names: Name of level to add.
    :param values: Value of the level.
    :returns: DataFrame with appended level.  
    '''
    df = pd.concat ({value: df}, names = [name], axis = axis)
    return df

def add_levels (df, values, names, axis = 0):
    '''
    Adds levels with a single value to the index of a DataFrame.
    :param names: List of names of levels to add.
    :param values: List of values for each level.
    :returns: DataFrame with appended levels.  
    '''
    for value, name in zip (values [ : : -1], names [ : : -1]):
        df = add_level (df, value, name, axis = axis)
    return df


# NumPy functions


def weighted_avg_and_std (values, weights):
    """
    Calculates weighted average and standard deviation.
    :param values: Array containing data to be averaged.
    :weights: Array of weights associated with the values.
    Should have the same shape as values.
    :returns: Weighted average and standard deviation.
    """
    average = np.average (values, weights = weights)
    variance = np.average ((values - average) ** 2, weights = weights)
    return (average, math.sqrt (variance))


# MatPlotLib functions


def set_axlims (series, margin_factor = 0.1):
    """
    Fix for a scaling issue with matplotlib's scatterplot and small values.
    To be used with .set_ylim (bottom, top)
    :param series: Series as given to plt.scatter.
    :param margin_factor: A fraction of data to expand the borders with.
    :returns: A tuple of limits (bottom, top)
    """
    minv = series.min ()
    maxv = series.max ()
    data_range = maxv - minv
    border = abs (data_range * margin_factor)
    maxlim = maxv + border
    minlim = minv - border

    return minlim, maxlim


def unique_legend (ax):
    '''
    Takes axis' handles and labels and returns uniques. 
    :param ax: MatPlotLib axis object.
    :returns: A dictionary indexed by handles and labels. 
    Can be directly unpacked into ax.legend ().
    '''
    handles, labels =  ax.get_legend_handles_labels()
    by_label = dict (zip (labels, handles))
    by_label = {
        'handles': by_label.values (),
        'labels':  by_label.keys ()
    }
    return by_label

