import math
import re
import os.path
import pandas as pd
import numpy as np
from scipy.integrate import simpson
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d


def change_extension(path, new_extension):
    '''
    Thanges path's extension.
    :param path: File path.
    :param new_extension: New extension.
    :returns: Path with new extension.
    '''
    base, _ = os.path.splitext(path)
    return f'{base}.{new_extension}'


def metadata_from_file_name(path, pattern, as_list=False):
    '''
    Takes file name from the full path and matches pattern.
    :param path: Full path to file.
    :param pattern: RegEx pattern.
    :returns: Matched groups.
    '''
    file_name = os.path.basename(path)
    base_name, _ = os.path.splitext(file_name)
    pattern = re.compile(pattern)

    match = re.match(pattern, base_name)
    if as_list:
        if not match:
            return match

        a = 1
        matches = []
        while True:
            try:
                matches.append(match.group(a))
                a += 1
            except IndexError:
                return matches

    return match

# Pandas functions


def add_level(df, value, name, axis=0):
    '''
    Adds level with a single value to the index of a DataFrame.
    :param df: DataFrame to append the level to.
    :param value: Value of the level.
    :param name: Name of level to add.
    :param axis: The axis to add the level to. [Default: 0]
    :returns: DataFrame with appended level.  
    '''
    df = df.copy()
    df = pd.concat({value: df}, names=[name], axis=axis)
    return df


def add_levels(df, values, names, axis=0):
    '''
    Adds levels with a single value to the index of a DataFrame.
    :param df: DataFrame to append the level to.
    :param names: List of names of levels to add.
    :param values: List of values for each level.
    :param axis: The axis to add the level to. [Default: 0]
    :returns: DataFrame with appended levels.  
    '''
    df = df.copy()
    for value, name in zip(values[:: -1], names[:: -1]):
        df = add_level(df, value, name, axis=axis)
    return df

def flatten_column_index (df, linker='_'):
    '''
    Flattens multi to a single index in df.
    :param df: Pandas DataFrame.
    :param linker: Symbol to use when linking level values.
    :returns: Pandas DataFrame with a single column index.
    '''
    df = df.copy()
    df.columns = [linker.join(col) for col in df.columns.values]
    return df


def export_pickle(df, path, **kwargs):
    '''
    Exports pickle.
    :param df: Pandas DataFrame.
    :param path: Path for new file.
    :param kwargs: Keyword arguments passed to `pd.to_pickle` function.
    '''
    pd.to_pickle(df, path, **kwargs)

# NumPy functions


def weighted_avg_and_std(values, weights):
    """
    Calculates weighted average and standard deviation.
    :param values: Array containing data to be averaged.
    :weights: Array of weights associated with the values.
    Should have the same shape as values.
    :returns: Weighted average and standard deviation.
    """
    average = np.average(values, weights=weights)
    variance = np.average((values - average) ** 2, weights=weights)
    return (average, math.sqrt(variance))


# SciPy functions

def interpolate(df, step, kind='linear'):
    '''
    Generates a new DataFrame with index step as defined in `step` parameter.
    Index min and max values remain the same.
    :param df: Pandas DataFrame.
    :param step: New index step size.
    :param kind: Specifies the kind of interpolation as a string or as an integer 
    specifying the order of the spline interpolator to use. The string has to be one of 
    ‘linear’, ‘nearest’, ‘nearest-up’, ‘zero’, ‘slinear’, ‘quadratic’, ‘cubic’, ‘previous’,
    or ‘next’. See scipy.interpolate.interp1d for more detail. [default: 'linear]
    :returns: Expanded and interpolated DataFrame
    '''
    df = df.sort_index()
    x = df.index
    xnew = np.arange(x.min(), x.max(), step)
    int_df = []
    for _, data in df.groupby(df.columns, axis=1):
        y = np.transpose(data.values)[0]
        f = interp1d(x, y, kind)
        int_data = pd.DataFrame(f(xnew), index=xnew, columns=data.columns)
        int_df.append(int_data)
    return pd.concat(int_df, axis=1)


def apply_savgol(df, window_length=25, polyorder=3, deriv=0, **kwargs):
    '''
    Applies a Savitzky-Golay filter to a DataFrame.
    :param df: DataFrame to be filtered.
    :param window_length: The length of the filter window. Must be a positive odd integer.
        [Default: 25]
    :param polyorder:  Order of the polynomial used to fit the samples. Must be less than window_length.
        [Default: 3]
    :param deriv: Order of the derivative to compute. 0 means to filter the data without differentiating. 
        [Default: 0]
    :param kwargs: Keyword arguments passed to scipy.signal.savgol_filter.
    :returns: DataFrame with filtered data.
    '''
    if not isinstance(df, pd.DataFrame):
        # change series to dataframe
        df = pd.DataFrame(df)

    def savgol(x):
        return savgol_filter(x, window_length=window_length, polyorder=polyorder, deriv=deriv, **kwargs)

    return df.apply(savgol)


def simpson_integrate(df, start=None, end=None):
    '''Applies Simpson integration to a DataFrame.
    :param df: Pandas DataFrame.
    :param start: Lower bound of the search range, or None. [Default: None]
    :param end: Upper bound of the search range, or None. [Default: None]
    :returns: Pandas Series with calculated integrals.
    '''
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    data = df.loc[start:end]
    data = data.dropna()
    return data.apply(simpson, x=data.index)


# MatPlotLib functions


def set_axlims(series, margin_factor=0.1):
    """
    Fix for a scaling issue with matplotlib's scatterplot and small values.
    To be used with .set_ylim (bottom, top)
    :param series: Series as given to plt.scatter.
    :param margin_factor: A fraction of data to expand the borders with.
    :returns: A tuple of limits (bottom, top)
    """
    minv = series.min()
    maxv = series.max()
    data_range = maxv - minv
    border = abs(data_range * margin_factor)
    maxlim = maxv + border
    minlim = minv - border

    return minlim, maxlim


def unique_legend(ax):
    '''
    Takes axis' handles and labels and returns uniques. 
    :param ax: MatPlotLib axis object.
    :returns: A dictionary indexed by handles and labels. 
    Can be directly unpacked into ax.legend ().
    '''
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    by_label = {
        'handles': by_label.values(),
        'labels':  by_label.keys()
    }
    return by_label
