import sys
import pandas as pd
import re
import os

from .. import bix_standard_functions as bsf
from .. import thot as bt


def get_substrate_name(path, pattern=r'[^\.]*'):
    '''
    Looks for substrate and pixel name. Splits the string around an underscore, if there is an underscore,
    the part after it is considered pixel name.
    :param path: File path.
    :param search: Search pattern for the part with the substrate and pixel name. [Default: r'[^\.]*']
    :returns: List with substrate name or substrate and pixel name.
    '''
    file_name = os.path.basename(path)
    match = re.search(pattern, file_name).group()
    match = match.split('_')
    if len(match) == 1:
        match.append('')
    elif len(match) > 2:
        raise ValueError(f'Wrong file name: {file_name}')
    return match


def append_substrate_meta(path, df, has_pixel=True, **kwargs):
    '''
    Appends substrate metadata to columns.
    :param path: File path.
    :param df: Pandas DataFrame.
    :param has_pixel: Whether pixel name should also be 
    appended. [Default: True]
    :param kwargs: Keyword arguments passed to get_substrate_name.
    :returns: Pandas DataFrame with appended metadata.
    '''
    match = get_substrate_name(path, **kwargs)
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


def import_raw_data(
    db,
    import_file,
    search={'type': ''},
    has_date=True,
    rename_axis=True,
    sort_columns=True,
    i_file_kwargs={},
    **kwargs
):
    '''
    Imports raw data from a database.
    :param db: ThotProject instance.
    :param import_file: Function that takes file path and
    outputs a pandas dataframe.
    :param search: Raw asset search pattern. [Default: {'type': ''}]
    :param has_pixel: Whether pixel name should be 
    appended. [Default: True]
    :param has_date: Whether date should be appended. [Default: True]
    :param rename_axis: Whether to rename column levels. [Default: True]
    :param sort_columns: Whether to sort column index. [Default: True]
    :param i_file_kwargs: Dictionary of keyword arguments passed to import_file. [Default: {}]
    :param kwargs: Keyword arguments passed to append_substrate_meta.
    :returns: Pandas DataFrame.
    '''
    assets = bt.find_assets(db, search)
    if has_date:
        date = get_date(db)
    files = [asset.file for asset in assets]
    files = sorted(files)
    dfs = []
    for file in files:
        df = import_file(file, **i_file_kwargs)
        if has_date:
            df = bsf.add_level(df, date, 'date', axis=1)
        df = append_substrate_meta(file, df, **kwargs)
        dfs.append(df)
    df = pd.concat(dfs, axis=1)
    if sort_columns:
        df = df.sort_index(axis=1)
    if rename_axis:
        df.columns = df.columns.set_names('param', level=-1)
    return df


def import_formatted_data(db, search, import_file=pd.read_pickle, axis=0, **kwargs):
    '''
    '''
    if not isinstance(search, list):
        search = [search]

    assets = []
    for s in search:
        found = bt.find_assets(db, search=s, exit=False)
        assets += found

    if assets == []:
        sys.exit()

    if len(assets) == 1:
        return import_file(assets[0].file, **kwargs)

    df = []
    for asset in assets:
        df.append(import_file(asset.file, **kwargs))
    return pd.concat(df, axis=axis)
