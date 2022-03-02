import sys
import pandas as pd
import re
import os

from .. import bix_standard_functions as bsf
from .. import thot as bt


def get_substrate_name(path):
    '''
    Takes file name and removes the part after the first dot.
    Splits the string around an underscore, if there is an underscore,
    the part after it is considered pixel name.
    :param path: File path.
    :returns: List with substrate name or substrate and pixel name.
    '''
    file_name = os.path.basename(path)
    match = re.search(r'[^\.]*', file_name).group()
    match = match.split('_')
    if len(match) == 1:
        match.append('')
    elif len(match) > 2:
        raise ValueError(f'Wrong file name: {file_name}')
    return match


def append_substrate_meta(path, df, has_pixel=True):
    '''
    Appends substrate metadata to columns.
    :param path: File path.
    :param df: Pandas DataFrame.
    :param has_pixel: Whether pixel name should also be 
    appended. [Default: True]
    :returns: Pandas DataFrame with appended metadata.
    '''
    match = get_substrate_name(path)
    if has_pixel:
        df = bsf.add_levels(df, match, ['substrate', 'pixel'], axis=1)
    else:
        df = bsf.add_level(df, match[0], 'substrate', axis=1)
    return df


def get_date(db):
    '''
    Gets date from container metadata.
    :param db: ThotProject instance.
    :returns: Timestamp or None if not defined.
    '''
    container = db.find_container({"_id": db.root})
    if 'date' in container.metadata:
        date = container.metadata['date']
        try:
            date = pd.to_datetime(date)
        except ValueError:
            raise ValueError(f'Wrong date format: {date}')
    else:
        date = ''
    return date


def import_raw_data(db, import_file, search={'type': ''}, has_pixel=True, **kwargs):
    '''
    Imports raw data from a database.
    :param db: ThotProject instance.
    :param import_file: Function that takes file path and
    outputs a pandas dataframe.
    :param search: Raw asset search pattern. [Default: {'type': ''}]
    :param has_pixel: Whether pixel name should also be 
    appended. [Default: True]
    :param kwargs: Keyword arguments passed to import_file.
    :returns: Pandas DataFrame.
    '''
    assets = bt.find_assets(db, search)
    date = get_date(db)
    dfs = []
    for asset in assets:
        df = import_file(asset.file, **kwargs)
        df = bsf.add_level(df, date, 'date', axis=1)
        df = append_substrate_meta(asset.file, df, has_pixel=has_pixel)
        dfs.append(df)
    df = pd.concat(dfs, axis=1)
    df = df.sort_index(axis=1)
    if has_pixel:
        df = df.rename_axis(columns=['substrate', 'pixel', 'date', 'param'])
    else:
        df = df.rename_axis(columns=['substrate', 'date', 'param'])
    return df


def import_formatted_data(db, search, import_file=pd.read_pickle, axis=0, **kwargs):
    '''
    '''
    if not isinstance(search, list):
        search = [search]

    assets = []
    for a_t in search:
        found = bt.find_assets(db, search={'type': a_t}, exit=False)
        assets += found

    if assets == []:
        sys.exit()

    if len(assets) == 1:
        return import_file(assets[0].file, **kwargs)

    df = []
    for asset in assets:
        df.append(import_file(asset.file, **kwargs))
    return pd.concat(df, axis=axis)
