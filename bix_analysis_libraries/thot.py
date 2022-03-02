import os.path
import sys
import pandas as pd


def find_assets(db, search={'type': ''}, exit=True):
    '''
    Finds raw assets within a thot container.
    :param db: A ThotProject instance.
    :param search: Asset search pattern. [Default: {'type': ''}]
    :param exit: Wheter to end the script when no assets are found. 
    [Default: True]
    :returns: List of matched assets.
    '''
    assets = db.find_assets(search)
    if assets == [] and exit:
        sys.exit()
    return assets


def export_asset(file, db, export_function, item, a_type=None, **kwargs):
    '''
    Exports an asset.
    :param file: File name.
    :param db: ThotProject instance.
    :param export_function: Function that takes in path and item to export.
    :param item: Item to export.
    :param a_type: Asset type. If `None`, obtained by removing file extension.
    [Default: None]  
    param kwargs: Keyword arguments passed to export_function. 
    '''
    no_extension, _ = os.path.splitext(file)
    if a_type is None:
        a_type = no_extension
    props = {
        'file': file,
        'type': a_type,
        'tags': a_type.split('_')
    }
    asset_path = db.add_asset(props, no_extension)
    export_function(item, asset_path, **kwargs)


def make_global_asset(db, a_type, a_path):
    '''
    Makes a global asset if such asset does not exist already.
    :param db: ThotProject instance.
    :param a_type: Asset type.
    :param a_path: Asset path.
    :returns: Asset instance.
    '''
    search = {'type': a_type}
    asset = db.find_asset(search)
    if asset is None:
        props = search.copy()
        props['file'] = a_path
        db.add_asset(props, a_type)
        asset = db.find_asset(search)
    return asset


def import_global_asset(
    db,
    a_type,
    a_path,
    dev_path=None,
    import_function=pd.read_pickle,
    **kwargs
):
    '''
    Creates a global asset if it doesn't exist and imports it.
    :param db: ThotProject instance.
    :param a_type: Asset type.
    :param a_path: Asset path.
    :param dev_path: Relative path from the script dir to the file. Used in dev mode.
    If None, asset_path is used. [Default: None]
    :param import_function: Function that takes file path as argument and imports it. 
    [Default: pd.read_pickle]
    :param kwargs: Keyword arguments passed to import_function.
    :returns: Object returned by import_function.
    '''
    if dev_path and db.dev_mode():
        path = os.environ['THOT_ORIGINAL_DIR']
        file = os.path.join(path, dev_path)
    else:
        asset = make_global_asset(db, a_type, a_path)
        file = asset.file
    return import_function(file, **kwargs)
