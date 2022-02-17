from plotly import graph_objects as go


def export_plotly(fig, path):
    '''
    Exports html figure.
    :param path: File path.
    :param fig: Plotly figure instance.
    '''
    fig.write_html(path, include_plotlyjs='cdn')


def multilayer_heatmap(df, layers, drop_pos=(0.37, 1.1)):
    '''
    Plots params as heatmap for data exploration.
    :param df: Pandas DataFrame indexed by y cordinate in Index, and x coordinate, as well as params in Columns.
    :param layers: List of params from Columns to plot.
    :param drop_pos: Tuple containing x and y position of the dropdown menu. [Default: (0.37, 1.1)]
    :returns: plotly.graph_objects.Figure instance.
    '''
    fig = go.Figure()

    def add_trace(fig, data, visible=True):
        fig.add_trace(
            go.Heatmap(
                x=data.columns.get_level_values(0).tolist(),
                y=data.index.values.tolist(),
                z=data,
                visible=visible
            )
        )

    lpr = len(layers)
    bl = []
    for (nr, param) in enumerate(layers):
        if nr == 0:
            add_trace(fig, df[param])
        else:
            add_trace(fig, df[param], visible=False)
        tfl = lpr * [False]
        tfl[nr] = True
        bl.append(
            dict(
                args=[{'visible': tfl}],
                label=param,
                method='update'
            )
        )
    fig.update_layout(
        updatemenus=[
            dict(
                buttons=bl,
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=drop_pos[0],
                xanchor="left",
                y=drop_pos[1],
                yanchor="top"
            ),
        ]
    )
    return fig


def plot_PL_params(df, params):
    '''
    Plots PL params as 3D surface and heatmap for data exploration.
    :param df: Pandas DataFrame indexed by y cordinate in Index, and x coordinate, as well as params in Columns.
    :param params: List of params from Columns to plot.
    :returns: plotly.graph_objects.Figure instance.
    '''
    fig = go.Figure()

    def add_trace(fig, data, visible=True):
        fig.add_trace(
            go.Heatmap(
                x=data.columns.get_level_values(0).tolist(),
                y=data.index.values.tolist(),
                z=data,
                visible=visible
            )
        )

    lpr = len(params)
    bl = []
    for (nr, param) in enumerate(params):
        if nr == 0:
            add_trace(fig, df[param])
        else:
            add_trace(fig, df[param], visible=False)
        tfl = lpr * [False]
        tfl[nr] = True
        bl.append(
            dict(
                args=[{'visible': tfl}],
                label=param,
                method='update'
            )
        )
    fig.update_layout(
        updatemenus=[
            dict(
                buttons=list([
                    dict(
                        args=["type", "heatmap"],
                        label="Heatmap",
                        method="restyle"
                    ),
                    dict(
                        args=["type", "surface"],
                        label="3D Surface",
                        method="restyle"
                    ),
                ]),
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.1,
                xanchor="left",
                y=1.1,
                yanchor="top"
            ),
            dict(
                buttons=bl,
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.37,
                xanchor="left",
                y=1.1,
                yanchor="top"
            ),
        ]
    )
    return (fig)
