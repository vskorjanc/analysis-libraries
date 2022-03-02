from plotly import graph_objects as go


def export_plotly(fig, path):
    '''
    Exports html figure.
    :param path: File path.
    :param fig: Plotly figure instance.
    '''
    fig.write_html(path, include_plotlyjs='cdn')


def heatmap_3D_plot(df, xaxis='x', params=None):
    '''
    Plots params as 3D surface and heatmap for data exploration.
    :param df: Pandas DataFrame with MultiIndex Index that contains x and y values.
    :param xaxis: X axis name (string) or position (integer). [Default: 'x']  
    :param params: List of params from Columns to plot. 
    If None, plots all the Params. [Default: None]
    :returns: plotly.graph_objects.Figure instance.
    '''
    if params is None:
        params = df.columns

    df = df.unstack(xaxis)

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
                # direction="down",
                type='buttons',
                direction='right',
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.37,
                xanchor="left",
                y=1.1,
                yanchor="top"
            ),
            dict(
                buttons=list([
                    dict(
                        args=[{
                            "scene.camera.eye": {'x': 0, 'y': 0, 'z': 2},
                            "scene.camera.up": {'x': 0, 'y': 1, 'z': 0},
                        }],
                        label="Rotate back",
                        method="relayout"
                    ),
                ]),
                type='buttons',
                pad={"r": 10, "t": 10},
                x=1,
                xanchor="right",
                y=1.1,
                yanchor="top"
            ),
            dict(
                buttons=bl,
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.1,
                xanchor="left",
                y=1.1,
                yanchor="top"
            ),
        ]
    )
    camera = dict(
        up=dict(x=0, y=1, z=0),
        eye=dict(x=0, y=0, z=2)
    )
    fig.update_layout(
        margin=dict(t=100, b=0, l=0, r=0),
        scene_camera=camera
    )
    return (fig)
