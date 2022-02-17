def find_assets(db, search={'type': ''}):
    '''
    Finds raw assets within a thot container.
    :param db: A ThotProject instance.
    :param ra_type: Asset search pattern. [Default: {'type': ''}]
    :returns: List of matched assets.
    '''
    return db.find_assets(search)
