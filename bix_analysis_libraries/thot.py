import os.path
import sys
import pandas as pd
from thot import ThotProject


def init_thot(
    script_path: str,
    dev_roots: str = '../../dev_roots.csv'
):
    '''
    Infers the `dev_root` from the list on dev_roots, and initializes `ThotProject`.
    :param script_path: Path to the script that runs (use __file__).
    :param dev_roots: Path pointing to a file that contains all the dev_roots.
    :returns: ThotProject instance
    '''
    def get_dev_root(script_name, dev_roots):
        df = pd.read_csv(dev_roots, sep='\t', header=0, index_col=0)
        return df.loc[script_name, 'dev_root']

    if 'THOT_ORIGINAL_DIR' in os.environ:
        original_dir = os.environ['THOT_ORIGINAL_DIR']
        os.chdir(original_dir)

    script_name = os.path.basename(script_path)
    try:
        dev_root = get_dev_root(script_name, dev_roots)
    except FileNotFoundError:
        dev_root = None
    return ThotProject(dev_root=dev_root)


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
    return asset_path


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
