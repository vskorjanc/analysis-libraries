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


def append_substrate_meta(path, df):
    '''
    Appends substrate metadata to columns.
    :param path: File path.
    :param df: Pandas DataFrame.
    :returns: Pandas DataFrame with appended metadata.
    '''
    match = get_substrate_name(path)
    df = bsf.add_levels(df, match, ['substrate', 'pixel'], axis=1)
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


def import_raw_data(db, import_file, search={'type': ''}, **kwargs):
    '''
    Imports raw data from a database.
    :param db: ThotProject instance.
    :param import_file: Function that takes file path and
    outputs a pandas dataframe.
    :param search: Raw asset search pattern. [Default: {'type': ''}]
    :param kwargs: Keyword arguments passed to import_file.
    :returns: Pandas DataFrame.
    '''
    assets = bt.find_assets(db, search)
    date = get_date(db)
    dfs = []
    for asset in assets:
        df = import_file(asset.file, **kwargs)
        df = bsf.add_level(df, date, 'date', axis=1)
        df = append_substrate_meta(asset.file, df)
        dfs.append(df)
    df = pd.concat(dfs, axis=1)
    df = df.sort_index(axis=1)
    df = df.rename_axis(columns=['substrate', 'pixel', 'date', 'param'])
    return df
