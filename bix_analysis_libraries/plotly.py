import pandas as pd
from plotly import graph_objects as go
from plotly import express as px


def export_plotly(fig, path):
    '''
    Exports html figure.
    :param path: File path.
    :param fig: Plotly figure instance.
    '''
    fig.write_html(path, include_plotlyjs='cdn')


def scatter_4D_plot(
    df: pd.DataFrame,
    xyz: tuple[str, str, str],
    color: str,
) -> go.Figure:
    """Makes a 4D scatter plot (3 spatial + color) with buttons for watching different planes.

    :param df: DataFrame with columns as 4 dimensions that are plotted. Columns should not be MultiIndex.
    :param xyz: Tuple of x, y and z axis column names.
    :param color: Color axis name.
    :return: Plotly figure.
    """
    x, y, z = xyz
    fig = px.scatter_3d(
        df,
        x=x,
        y=y,
        z=z,
        # size='area',
        color=color
    )
    fig.update_traces(
        marker=dict(size=3),
        selector=dict(mode='markers')
    )

    def make_button(eye, label, x_pos):
        button = dict(
            buttons=list([
                dict(
                    args=[{
                        "scene.camera.eye": eye,
                        "scene.camera.up": {'x': 0, 'y': 1, 'z': 0},
                    }],
                    label=label,
                    method="relayout"
                ),
            ]),
            type='buttons',
            pad={"r": 10, "t": 10},
            showactive=True,
            x=x_pos,
            xanchor="right",
            y=1.1,
            yanchor="top"
        )
        return button

    def make_buttons(x, y, z):
        xy = make_button(
            {'x': 0, 'y': 0, 'z': 2},
            f'{x}-{y}',
            x_pos=0.6
        )
        zx = make_button(
            {'x': 0, 'y': 2, 'z': 0},
            f'{z}-{x}',
            x_pos=0.8
        )
        zy = make_button(
            {'x': 2, 'y': 0, 'z': 0},
            f'{z}-{y}',
            x_pos=1
        )
        return [xy, zx, zy]

    fig.update_layout(
        updatemenus=make_buttons(x, y, z)
    )
    return fig


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
        scene_camera=camera,
        xaxis_title=xaxis,
        yaxis_title=df.index.name
    )
    return (fig)


def multilayer_plot(
    df: pd.DataFrame,
    plot_single: go.Figure,
    params=None
):
    '''
    Takes 
    :param df: Pandas DataFrame with columns as layers.
    :plot single: Function that returns object passed to Plotly's fig.add_trace() function.
    Takes one column of the dataframe (layer) and `visible` as arguments. 
    :param params: List of params from Columns to plot. 
    If None, plots all the Params. [Default: None]
    :returns: plotly.graph_objects.Figure instance.
    '''
    if params is None:
        params = df.columns

    fig = go.Figure()

    def add_trace(fig, data, visible=True):
        fig.add_traces(plot_single(data, visible=visible))

    param_len = len(params)
    bl = []
    for (nr, param) in enumerate(params):
        if nr == 0:
            add_trace(fig, df[param])
            trace_len = len(fig.data)
        else:
            add_trace(fig, df[param], visible=False)
        is_visible = param_len * trace_len * [False]
        for i in range(trace_len):
            is_visible[nr * trace_len + i] = True
        bl.append(
            dict(
                args=[{'visible': is_visible}],
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
                x=0.1,
                xanchor="left",
                y=1.1,
                yanchor="top"
            ),
        ]
    )
    # fig.update_layout(
    #     xaxis_title=xaxis,
    #     yaxis_title=df.index.name
    # )
    return (fig)
