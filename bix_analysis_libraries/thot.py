import os.path
from re import search


def find_assets(db, search={'type': ''}):
    '''
    Finds raw assets within a thot container.
    :param db: A ThotProject instance.
    :param ra_type: Asset search pattern. [Default: {'type': ''}]
    :returns: List of matched assets.
    '''
    return db.find_assets(search)


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


def make_global_asset(db, a_path, a_type):
    search = {'type': a_type}
    asset = db.find_asset(search)
    if asset is None:
        props = search.copy()
        props['file'] = a_path
        db.add_asset(props, a_type)
        asset = db.find_asset(search)
    return asset


def import_global_asset(db, a_path, a_type, import_function, dev_path=None, **kwargs):
    '''
    :param db: ThotProject instance.
    :param a_path: Asset path.
    :param a_type: Asset type.
    :param import_function: Function used to import the asset.
    :param dev_path: Relative path from the script dir to the file. Used in dev mode.
    If none, asset_path is used. [Default: None]
    :param import_function: Function that takes file path as argument and imports it. 
    :param kwargs: Keyword arguments passed to import_function.
    :returns: Object returned by import_function.
    '''
    if dev_path and db.dev_mode():
        path = os.environ['THOT_ORIGINAL_DIR']
        file = os.path.join(path, dev_path)
    else:
        asset = make_global_asset(db, a_path, a_type)
        file = asset.file
    return import_function(file, **kwargs)
