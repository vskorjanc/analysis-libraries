import logging
import os
from importlib import reload

# std libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# custom libraries
from bric_analysis_libraries import standard_functions as std
from bric_analysis_libraries.jv import ec_lab_data_prep as ecdp
from bric_analysis_libraries.jv import biologic_data_prep as bldp
from bric_analysis_libraries.jv import jv_analysis as jva


def import_mpp(
    folder,
    file_name='mpp.csv',
    skiprows=1
):
    '''
    Imports MPP data
    :param folder: Data path.
    :param file_name: File name. [Default = mpp.csv]
    :param skiprows: Number of rows to skip. [Default = 0]
    :returns: A Pandas DataFrame indexed by channel.
    '''
    logging.info('Importing data.')

    path = os.path.join(folder, file_name)

    # importing data
    df = pd.read_csv(path, header=[0, 1], skiprows=range(2, 2 + skiprows))
    logging.info('Data imported from "{}".'.format(path))

    # removing the Cycle column
    df = df.drop('Cycle', axis=1, level=1)

    # changing the columns' names to 'time', 'voltage', 'current' and 'power'
    channels = [int(a) for a in df.columns.get_level_values(0).unique()]

    metric = ['time', 'voltage', 'current', 'power']
    if 'Lamp' in df.columns.get_level_values(1):
        metric = ['time', 'voltage', 'current', 'power', 'lamp']
    df.columns = pd.MultiIndex.from_product(
        [channels, metric], names=['channel', 'parameter'])

    df = df.stack('channel').droplevel(0).sort_values(['channel', 'time'])

    # changing the sign of the 'power' column
    df['power'] = - df['power']

    return df


def import_mpps(
    folder,
    file_format='mpp-<>',
    skiprows=1
):
    '''
    Imports MPP data for a series of measurement with varied hold voltage.
    :param folder: Data path.
    :param file_format: File format, the experiment number is substituted
    by <>.  [Default = mpp-<>]
    :param skiprows: Number of rows to skip. [Default = 1]
    :returns: A Pandas DataFrame indexed by experiment and channel.
    '''
    files = os.listdir(folder)
    dfs = []

    for file in files:
        df = import_mpp(
            folder=folder,
            file_name=file,
            skiprows=skiprows,
        )
        ms = std.metadata_from_file_name(
            file_format,
            os.path.join(folder, file),
            is_numeric=True
        )
        std.insert_index_levels(
            df,
            int(ms),
            names='measurement',
            axis=0,
            inplace=True
        )
        dfs.append(df)

    dfs = pd.concat(dfs)
    return dfs


def add_hold_voltage(
    df,
    folder,
    file_name='hold_voltage.pkl'
):
    '''
    :param df: A Pandas dataframe indexed by measurement and channel.
    :param folder: Data path.
    :param file_name: Name of the .pkl file containing the voltage information. 
    [Default = 'hold_voltage.pkl'] 
    :returns: A Pandas dataframe with "hold_voltage" as an additional index level.
    '''
    logging.info('Adding hold voltage.')
    hv = pd.read_pickle(os.path.join(folder, file_name))
    df['hold_voltage'] = hv.stack().reorder_levels(['measurement', 'channel'])
    return df.set_index('hold_voltage', append=True)


def assign_cycles(
    df,
    mpl=0.01,
    bins=10,
    window=31
):
    '''
    Assigns a cycle column to a DataFrame.
    :param df: A pandas DataFrame.
    :param mpl: A factor to multiply the maximum value of counts to
    obtain the cutoff value. [Default: 0.01] 
    :param bins: Number of divisions in the histogram. [Default: 10]
    :param window: Size of window for function rolling ().median (),
    applied to 'light' column. Must be an odd natural number. [Default: 31]
    :returns: A Pandas DataFrame with cycle column. 
    '''

    logging.info('Appending cycle count.')
    df = df.copy()

    if 'lamp' not in df.columns.get_level_values('parameter'):
        (density, ranges) = np.histogram(
            df['power'].rolling(8).median().dropna(), bins=bins)

        cutoff = mpl * density.max()

        # threshold value taken after the lower max is reached,
        # when density is lower than the cutoff value (0 if no value determined)
        threshold = 0
        was_bigger = False
        for (n, d) in enumerate(density):
            if was_bigger == True and d < cutoff:
                threshold = ranges[n]
                break
            elif d > cutoff and was_bigger == False:
                was_bigger = True

        # light == 1 if power above threshold value
        df['lamp'] = np.where(df['power'] > threshold, 1, 0)
        df['lamp'] = df['lamp'].rolling(
            window, min_periods=1, center=True).median()

    # increase the cycle count when the light switches
    cycles = []
    cycle = 0
    last_light = 1
    for light in df['lamp'].dropna():
        if light != last_light:
            cycle += 1
        cycles.append(cycle)
        last_light = light
    df['cycle'] = cycles
    df = df.drop(columns='lamp')
    df = df.set_index('cycle', append=True)

    return df


def remove_odd_cycles(df):
    '''
    Removes data for odd cycles.
    :param df: A Pandas DataFrame indexed by cycles.
    :returns: A Pandas DataFrame without data for odd cycles.
    '''

    logging.info('Removing odd cycles')
    rdf = []
    cygb = df.groupby('cycle')
    for cy, data in cygb:
        if cy % 2 == 0:
            rdf.append(data)

    logging.info('Odd cycles removed.')

    return pd.concat(rdf)


def relative_time(df):
    '''
    Calculates the time since a cycle started.
    :param df: A Pandas DataFrame.
    :returns: A Pandas DataFrame with an additional column 'r_time'.
    '''

    logging.info('Calculating relative time.')
    dfc = df.copy()
    dfc['r_time'] = df.groupby('cycle')['time'].transform(
        lambda x: x - x.iloc[0])

    return dfc


def fit_d_exp(df, **kwargs):
    '''
    Fits an exponential function to the data.
    :param df: A Pandas DataFrame indexed by relative time.
    :param kwargs: Additional arguments passed to std.df_fit_function.
    :returns: A Pandas DataFrame of the fit parameters and std deviation
    '''

    def exp(t, P0, P1, P2, tau1, tau2): return (
        P0 + P1 * (1 - np.exp(-t/tau1)) + P2 * (1 - np.exp(-t/tau2))
    )

    # defining the guess parameters
    def guess(data):
        # define init, mid, and end point for P guesses
        p_i = data.iloc[1: 4].mean()
        p_m = data.iloc[50: 58].mean()
        p_e = data.iloc[-9: -1].mean()
        P0 = p_i
        P1 = p_m - p_i
        P2 = p_e - p_m
        tau1 = 12
        tau2 = 150
#         print (f'Guess: A = {A:.2E},\t t = {t:.2E},\t B = {B:.2E},\t C = {C:.2E}\t')
        return P0, P1, P2, tau1, tau2

    fit = std.df_fit_function(exp, guess=guess, **kwargs)
    return fit(df)


def min_max(df):
    '''
    Determines the maximal and minimal (absolute) power values for each cycle (smoothed out
    with a rolling median.
    :param df: A Pandas DataFrame with columns indexed by channel and rows by cycle.

    '''
    mm = []
    for params, data in df.groupby(['measurement', 'channel', 'hold_voltage', 'cycle']):
        a = data['power'].dropna().rolling(8).median()
        d = {
            'p_min': a.min(),
            'p_max': a.max()
        }
        idx = pd.MultiIndex.from_product([[p] for p in params], names=[
                                         'measurement', 'channel', 'hold_voltage', 'cycle'])
        p = pd.DataFrame(d, index=idx)
        mm.append(p)
    mm = pd.concat(mm, axis=0)
    return mm


# def remove_fit (df):
#     '''
#     Removes a value from the fit function if the ratio of the
#     corresponding std and the value is bigger than 0.2.
#     :param df: A Pandas DataFrame obtained by using fit_exp function.
#     :returns: A DataFrame in which the erroneous fit values were substituted by NaN.
#     '''

#     cpgb = df.groupby (level = ['channel', 'parameter'], axis = 1)
#     fit = []
#     for name, data in cpgb:
#         d = data [(*name, 'value')].mask (abs (data[(*name, 'value')]) < (data[(*name, 'std')] / 0.2))
#         fit.append (d)
#     fit = pd.concat (fit, axis = 1)
#     return fit

def import_jv(path):
    '''
    Imports JV scan data.
    :param path: Path to the .csv file.
    :returns: A DataFrame indexed by channel and voltage.'''
    df = pd.read_csv(path, header=[0, 1])
    df = df.stack(0)
    df.columns = ['current', 'power', 'voltage']
    df = df.set_index('voltage', append=True)
    df = df.droplevel(0)
    return df.sort_index()


def get_mpp(df):
    """
    Gets the maximum power point

    :param df: A Pandas DataFrame containing JV scans to evaluate
    :returns: A Pandas DataFrame with Vmpp with Jmpp and Pmpp
    """
    pmpp = pd.Series(df.power.min())          # Pmpp
    vmpp = pd.Series(df.power.idxmin())       # Vmpp
    jmpp = pd.Series(pmpp / vmpp)               # Jmpp

    return pd.concat((vmpp, jmpp, pmpp), keys=['vmpp', 'jmpp', 'pmpp'], axis=1)


def get_jsc(df):
    """
    Get the short circuit current

    :param df: A Pandas DataFrame containing JV sweeps
    :returns: A Pandas Series of short circuit currents
    """
    jsc = pd.Series(df.current.loc[df.index.values.max()])
    return jsc.rename('jsc')


def get_voc(df):
    """
    Get the open circuit voltage

    :param df: A Pandas DataFrame containing JV sweeps
    :returns: A Pandas Series of open circuit voltages
    """
    voc = pd.Series(df.index.values.min())
    return voc.rename('voc')


def get_jv_metrics(df):
    """
    Creates a Pandas DataFrame containing metric about JV curves
    Metrics include maximum power point (vmpp, jmpp, pmpp), open circuit voltage,
    short circuit current, fill factor, and efficiency

    :params df: The DataFrame containing the JV curves
    :returns: A Pandas DataFrame containing information about the curves
    """
    metrics = [get_mpp(df), get_voc(df), get_jsc(df)]
    metrics = pd.concat(metrics, axis=1)
    metrics = metrics.assign(ff=lambda x: x.pmpp / (x.voc * x.jsc))
    metrics.columns.name = 'metric'
    return metrics
