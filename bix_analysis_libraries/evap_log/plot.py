def add_vrect(fig, start, end, row=1, col=1):
    fig.add_vrect(
        x0=start,
        x1=end,
        fillcolor="LightSalmon",
        opacity=0.5,
        layer="below",
        line_width=0,
        row=row,
        col=col,
    )


def get_shutter_intervals(shut):
    """
    Creates a list of tuples which indicated when to plot that the substrate shutter was open.
    :param shut: Pandas series which contains shutter column indexed by time. Index must be sorted.
    :returns: List of tuples.
    """
    shut = shut.interpolate(method="nearest")
    switch = shut.diff().abs()
    timestamp = switch.index[switch == 2].to_list()
    if shut.iloc[0] == 2:
        start = shut.index.min()
        timestamp.insert(0, start)
    if shut.iloc[-1] == 2:
        end = shut.index.max()
        timestamp.append(end)
    len_tsp = len(timestamp)
    if len_tsp % 2 == 1:
        raise ValueError("Shutter opening incorrectly calculated")
    pair_number = int(len_tsp / 2)
    open_close = [(timestamp[i * 2], timestamp[i * 2 + 1]) for i in range(pair_number)]
    return open_close


def show_shutter(shut, fig, row=1, col=1):
    """
    Adds rectangles with shutter open intervals to a plotly figure.
    """
    open_close = get_shutter_intervals(shut)
    for open, close in open_close:
        add_vrect(fig, open, close, row, col)
