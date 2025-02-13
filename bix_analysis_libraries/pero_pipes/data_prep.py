import sys
import pandas as pd
import re
import os

from .. import bix_standard_functions as bsf
from .. import thot as bt


def get_substrate_name(path, pattern=r"([^_]*?)(?:_([a-f]))?\..*"):
    """
    Looks for substrate and pixel name. Splits the string around an underscore, if there is an underscore,
    the part after it is considered pixel name.
    :param path: File path.
    :param search: Search pattern for the part with the substrate and pixel name. [Default: r'[^\.]*']
    :returns: List with substrate name or substrate and pixel name.
    """
    file_name = os.path.basename(path)
    match = re.match(pattern, file_name).groups()
    if len(match) > 2 or len(match) == 0:
        raise ValueError(f"Wrong file name: {file_name}")
    if len(match) == 2:
        if match[1] == None:
            match = [match[0]]
    return match


def append_substrate_meta(path, df, **kwargs):
    """
    Appends substrate metadata to columns.
    :param path: File path.
    :param df: Pandas DataFrame.
    :param kwargs: Keyword arguments passed to get_substrate_name.
    :returns: Pandas DataFrame with appended metadata.
    """
    match = get_substrate_name(path, **kwargs)
    if len(match) == 2:
        df = bsf.add_levels(df, match, ["substrate", "pixel"], axis=1)
    else:
        df = bsf.add_level(df, match[0], "substrate", axis=1)
    return df


def get_date(db):
    """
    Gets date from container metadata.
    :param db: ThotProject instance.
    :returns: Timestamp or None if not defined.
    """
    container = db.find_container({"_id": db.root})
    if "date" in container.metadata:
        date = container.metadata["date"]
        try:
            date = pd.to_datetime(date)
        except ValueError:
            raise ValueError(f"Wrong date format: {date}")
    else:
        date = ""
    return date


def import_raw_data(
    db,
    import_file,
    search={"type": ""},
    has_date=True,
    rename_axis=True,
    sort_columns=True,
    extension=None,
    i_file_kwargs={},
    **kwargs,
):
    """
    Imports raw data from a database.
    :param db: ThotProject instance.
    :param import_file: Function that takes file path and
    outputs a pandas dataframe.
    :param search: Raw asset search pattern. [Default: {'type': ''}]
    :param has_date: Whether date should be appended. [Default: True]
    :param rename_axis: Whether to rename column levels. [Default: True]
    :param sort_columns: Whether to sort column index. [Default: True]
    :param extension: Filter for files with the extension. [Default: None]
    :param i_file_kwargs: Dictionary of keyword arguments passed to import_file. [Default: {}]
    :param kwargs: Keyword arguments passed to append_substrate_meta.
    :returns: Pandas DataFrame.
    """
    assets = bt.find_assets(db, search)
    if has_date:
        date = get_date(db)
    files = [asset.file for asset in assets]
    if extension:
        if extension[0] != ".":
            extension = f".{extension}"
        files = [file for file in files if extension in file]
    files = sorted(files)
    dfs = []
    for file in files:
        df = import_file(file, **i_file_kwargs)
        if df.empty:
            continue
        if has_date:
            df = bsf.add_level(df, date, "date", axis=1)
        df = append_substrate_meta(file, df, **kwargs)
        dfs.append(df)
    df = pd.concat(dfs, axis=1)
    if sort_columns:
        df = df.sort_index(axis=1)
    if rename_axis:
        df.columns = df.columns.set_names("param", level=-1)
    return df


def import_formatted_data(db, search, import_file=pd.read_pickle, axis=0, **kwargs):
    """
    :param db: ThotProject instance.
    :param search: Asset search pattern.
    :param import_file: Function that takes file path and
    outputs a pandas dataframe. [Default: pd.read_pickle]
    :param axis: Axis along which to concatenate the imported dataframes. [Default: 0]
    :param kwargs: Keyword arguments passed to import_file function.
    :returns: Pandas DataFrame.
    """
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


def pickle_w_markdown(df, name, db, floatfmt=".2E", **kwargs):
    """
    Function that exports an .md together with a .pkl file. Intended for showing exported metrics in Obsidian.
    :param df: DataFrame to export.
    :param name: File name without extension.
    :param db: ThotProject instance.
    :param floatfmt: Format in which to render numbers. [Default: ".2E"]
    :param kwargs: KeyWord arguments passed to bt.export_asset.
    """
    asset_path = bt.export_asset(
        file=f"{name}.pkl", db=db, export_function=bsf.export_pickle, item=df, **kwargs
    )
    markdown_path = bsf.change_extension(asset_path, "md")
    df.to_markdown(markdown_path, floatfmt=floatfmt)
