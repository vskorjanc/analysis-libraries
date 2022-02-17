import os.path


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
