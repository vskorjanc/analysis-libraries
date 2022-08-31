import pandas as pd
from plotly import graph_objects as go
from plotly import (
    express as px,
    io as pio
)


def export_plotly(fig, path, **kwargs):
    '''
    Exports html figure.
    :param path: File path.
    :param fig: Plotly figure instance.
    :param kwargs: Keyword arguments passed to fig.write_html.
    '''
    config = {
        'toImageButtonOptions': {
            'format': 'svg',  # one of png, svg, jpeg, webp
        },
        'displaylogo': False
    }
    fig.write_html(path, config=config, include_plotlyjs='cdn', **kwargs)


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
    if type(df) == pd.Series:
        params = ['']
    elif params is None:
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
    if lpr == 1:
        add_trace(fig, df)
    else:
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
                    buttons=bl,
                    direction="down",
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=0.1,
                    xanchor="left",
                    y=1.1,
                    yanchor="top"
                )
            ])

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
    params=None,
    **kwargs
):
    '''
    Takes 
    :param df: Pandas DataFrame with columns as layers.
    :param plot_single: Function that returns object passed to Plotly's fig.add_trace() function.
    Takes one column of the dataframe (layer) and `visible` as arguments. 
    :param params: List of params from Columns to plot. 
    If None, plots all the Params. [Default: None]
    :param kwargs: Keyword arguments passed to plot_single function.
    :returns: plotly.graph_objects.Figure instance.
    '''
    if params is None:
        params = df.columns

    fig = go.Figure()

    def add_trace(fig, data, visible=True):
        fig.add_traces(plot_single(data, visible=visible, **kwargs))

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
    return (fig)

# set the template


pio.templates['bix'] = go.layout.Template(
    layout={
        'annotationdefaults': {'arrowhead': 0, 'arrowwidth': 1},
        'autotypenumbers': 'strict',
        'coloraxis': {'colorbar': {'outlinewidth': 0, 'tickcolor': 'rgb(36,36,36)', 'ticks': 'outside'}},
        'colorscale': {'diverging': [[0.0, 'rgb(103,0,31)'], [0.1, 'rgb(178,24,43)'],
                                     [0.2, 'rgb(214,96,77)'], [0.3,
                                                               'rgb(244,165,130)'], [0.4, 'rgb(253,219,199)'],
                                     [0.5, 'rgb(247,247,247)'], [0.6,
                                                                 'rgb(209,229,240)'], [0.7, 'rgb(146,197,222)'],
                                     [0.8, 'rgb(67,147,195)'], [0.9,
                                                                'rgb(33,102,172)'], [1.0, 'rgb(5,48,97)']],
                       'sequential': 'Magma',
                       'sequentialminus': 'Sunsetdark'},
        'colorway': ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52'],
        # 'toImageButtonOptions': {
        # 'format': 'svg',  # one of png, svg, jpeg, webp
        # 'filename': 'custom_image',
        # 'scale': 1  # Multiply title/legend/axis/canvas sizes by this factor
        # },
        'font': {'color': 'rgb(0,0,0)',
                 'family': 'Helvetica,Futura,Calibri,Times New Roman',
                 'size': 18},
        'geo': {'bgcolor': 'white',
                'lakecolor': 'white',
                'landcolor': 'white',
                'showlakes': True,
                'showland': True,
                'subunitcolor': 'white'},
        'hoverlabel': {'align': 'left'},
        'hovermode': 'closest',
        'legend': {
            'bgcolor': 'rgba(255,255,255,0.7)',
            'xanchor': 'left',
            'yanchor': 'top',
            'x': 0.01,
            'y': 0.99
        },
        'mapbox': {'style': 'light'},
        'paper_bgcolor': 'white',
        'plot_bgcolor': 'white',
        'polar': {'angularaxis': {'gridcolor': 'rgb(232,232,232)',
                                  'linecolor': 'rgb(36,36,36)',
                                  'showgrid': False,
                                  'showline': True,
                                  'ticks': 'outside'},
                  'bgcolor': 'white',
                  'radialaxis': {'gridcolor': 'rgb(232,232,232)',
                                 'linecolor': 'rgb(36,36,36)',
                                 'showgrid': False,
                                 'showline': True,
                                 'ticks': 'outside'}},
        'scene': {'xaxis': {'backgroundcolor': 'white',
                            'gridcolor': 'rgb(232,232,232)',
                            'gridwidth': 2,
                            'linecolor': 'rgb(36,36,36)',
                            'showbackground': True,
                            'showgrid': False,
                            'showline': True,
                            'ticks': 'outside',
                            'zeroline': False,
                            'zerolinecolor': 'rgb(36,36,36)'},
                  'yaxis': {'backgroundcolor': 'white',
                            'gridcolor': 'rgb(232,232,232)',
                            'gridwidth': 2,
                            'linecolor': 'rgb(36,36,36)',
                            'showbackground': True,
                            'showgrid': False,
                            'showline': True,
                            'ticks': 'outside',
                            'zeroline': False,
                            'zerolinecolor': 'rgb(36,36,36)'},
                  'zaxis': {'backgroundcolor': 'white',
                            'gridcolor': 'rgb(232,232,232)',
                            'gridwidth': 2,
                            'linecolor': 'rgb(36,36,36)',
                            'showbackground': True,
                            'showgrid': False,
                            'showline': True,
                            'ticks': 'outside',
                            'zeroline': False,
                            'zerolinecolor': 'rgb(36,36,36)'}},
        'shapedefaults': {'fillcolor': 'black', 'line': {'width': 0}, 'opacity': 0.3},
        'ternary': {'aaxis': {'gridcolor': 'rgb(232,232,232)',
                              'linecolor': 'rgb(36,36,36)',
                              'showgrid': False,
                              'showline': True,
                              'ticks': 'outside'},
                    'baxis': {'gridcolor': 'rgb(232,232,232)',
                              'linecolor': 'rgb(36,36,36)',
                              'showgrid': False,
                              'showline': True,
                              'ticks': 'outside'},
                    'bgcolor': 'white',
                    'caxis': {'gridcolor': 'rgb(232,232,232)',
                              'linecolor': 'rgb(36,36,36)',
                              'showgrid': False,
                              'showline': True,
                              'ticks': 'outside'}},
        'title': {'x': 0.5,
                  'y': 0.9,
                  'xanchor': 'center',
                  'yanchor': 'top'},
        'xaxis': {'automargin': True,
                  'gridcolor': 'rgb(232,232,232)',
                  'linecolor': 'rgb(36,36,36)',
                  'showgrid': True,
                  'showline': True,
                  'ticks': 'outside',
                  'title': {'standoff': 15},
                  'zeroline': True,
                  'zerolinecolor': 'rgb(170,170,170)',
                  'mirror': True},
        'yaxis': {'automargin': True,
                  'gridcolor': 'rgb(232,232,232)',
                  'linecolor': 'rgb(36,36,36)',
                  'showgrid': True,
                  'showline': True,
                  'ticks': 'outside',
                  'title': {'standoff': 15},
                  'zeroline': True,
                  'zerolinecolor': 'rgb(170,170,170)',
                  'mirror': True}
    })

pio.templates.default = "bix"
