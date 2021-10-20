#!/usr/bin/env python
# coding: utf-8

# In[1]:


# standard imports
import pandas as pd
import numpy as np
from itertools import groupby


def extract_temp (path):
    '''
    Extracts information about applied temperatures from a temp_track file.
    :param path: Path pointing to the temp_track.csv file.
    :returns: A list of applied temperatures.
    '''
    temperatures = pd.read_csv (
        path,
        sep = '\t',
        names = ['datetime', 'time', 'set_point', 'temperature']
    )
    temp_list = [int (k) for k, g in groupby (temperatures.set_point.values)]
        
    return temp_list

def define_dirn (temp_list):
    '''
    Defines in which direction the temperature is changing.
    :param temp_list: List of temperatures in the order they are applied.
    :returns: A list containing "down" or "up" as elements.
    '''
    dirn = []
    for index, temp in enumerate (temp_list):
        try: 
            dirn.append ('up' if temp < temp_list [index + 1] else 'down')
        except IndexError: 
            dirn.append ('up' if temp > temp_list [index -1] else 'down')
    
    return dirn

def add_temp (
    df, 
    temps,
    cutoff = 50
):
    '''
    Adds information about temperature and direction to the MPP dataframe.
    :param df: A pandas DataFrame.
    :param temps: A list of temperatures used in the measurement.
    :param cutoff: Cut off value, the minimal difference between times of two readings
    in seconds to increase the measurement count. [Default: 10]
    :returns: A pandas DataFrame indexed by temperature and direction.
    '''

    dfc = df.copy ()

    l = range (len (df))
    end = max (l)
    dirns = define_dirn (temps)
    dirn = []
    temp = []

    ms = 0
    for n in l:
        temp.append (temps [ms])
        dirn.append (dirns [ms])

        # increase the measurement count if time between two readings
        # is higher than the cutoff value
        if n != end and (df.time [n + 1] - df.time [n]) > cutoff:
            ms += 1

    dfc ['temperature'] = temp
    dfc ['direction']   = dirn
    return dfc.set_index (['temperature', 'direction'])

