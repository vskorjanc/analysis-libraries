import pandas as pd
import re
import os

from .. import bix_standard_functions as bsf
from .. import thot as bt


def get_substrate_name(path):
    file_name = os.path.basename(path)
    match = re.search(r'[^\.]*', file_name).group()
    match = match.split('_')
    if len(match) == 1:
        match.append('')
    elif len(match) > 2:
        raise ValueError(f'Wrong file name: {file_name}')
    return match


def append_substrate_meta(path, df):
    match = get_substrate_name(path)
    df = bsf.add_levels(df, match, ['substrate', 'pixel'], axis=1)
    return df


def get_date(thot):
    container = thot.find_container()
    if 'date' in container.metadata:
        date = container.metadata['date']
        try:
            date = pd.to_datetime(date)
        except ValueError:
            raise ValueError(f'Wrong date format: {date}')
    else:
        date = ''
    return date


def import_raw_data(thot, import_file):
    assets = bt.find_raw_assets(thot)
    date = get_date(thot)
    dfs = []
    for asset in assets:
        df = import_file(asset.file)
        df = bsf.add_level(df, date, 'date', axis=1)
        df = append_substrate_meta(asset.file, df)
        dfs.append(df)
    df = pd.concat(dfs, axis=1)
    df = df.sort_index(axis=1)
    df = df.rename_axis(columns=['substrate', 'pixel', 'date', 'param'])
    return df
