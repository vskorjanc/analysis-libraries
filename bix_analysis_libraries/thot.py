def find_raw_assets(thot, type=""):
    '''
    Finds raw assets within a thot container.
    :param thot: A ThotProject instance.
    :param type: Type search pattern. [default: ""]
    :returns: List of matched assets.
    '''
    return thot.find_assets({"type": type})


def find_assets(thot, type=""):
    '''
    Finds raw assets within a thot container.
    :param thot: A ThotProject instance.
    :param type: Type search pattern.
    :returns: List of matched assets.
    '''
    return thot.find_assets({"type": type})
