import pandas as pd
from . import bix_standard_functions as bsf
from plotly import graph_objects as go
from plotly import express as px, io as pio
import os.path


def update_fig(fig, fig_props):
    def update_trace(t):
        name = t.name
        if "_" in name:
            name = name.split("_")[0]
        t.update(
            name=fig_props.loc[name, "group"],
            line_color=fig_props.loc[name, "color"],
            legendrank=fig_props.loc[name, "legendrank"] + 1,
        )

    fig.for_each_trace(update_trace)


def export_plotly(
    fig,
    path,
    rename=False,
    fig_props_path="../JV/fig_props/fig_props.pkl",
    add_svg=False,
    add_png=False,
    dimensions=[600, 500],
    update_layout_kwargs={},
    **kwargs,
):
    if rename and os.path.isfile(fig_props_path):
        fig_props = pd.read_pickle(fig_props_path)
        update_fig(fig, fig_props)
    export_html(fig=fig, path=path, **kwargs)
    if not (add_png or add_png):
        return
    fig.update_layout(width=dimensions[0], height=dimensions[1], **update_layout_kwargs)
    if add_svg:
        path = bsf.change_extension(path, "svg")
        fig.write_image(path)
    if add_png:
        path = bsf.change_extension(path, "png")
        fig.write_image(path)


def export_html(fig, path, **kwargs):
    """
    Exports html figure.
    :param path: File path.
    :param fig: Plotly figure instance.
    :param kwargs: Keyword arguments passed to fig.write_html.
    """
    config = {
        "toImageButtonOptions": {
            "format": "svg",  # one of png, svg, jpeg, webp
        },
        "displaylogo": False,
    }
    fig.write_html(path, config=config, include_plotlyjs="cdn", **kwargs)


def give_next_color(color_count):
    """
    Returns Plotly color based on the color count.
    :param color_count: Counter which determines the color used.
    :returns: tuple of color, color_count. The color count is increased by one.
    """
    colors = px.colors.qualitative.Plotly
    color = colors[color_count % 10]
    color_count += 1
    return (color, color_count)


def add_log_button(
    fig: go.Figure, primary_log: bool = True, x: float = 0.11, y: float = 1.1
):
    """
    Adds a button that switches between linear and log mode for y axis (or both y and y2 axes if present).
    :param fig: Plotly figure.
    :param primary_log: Whether figure uses log view as primary. [Default: True]
    :param x: x position of the button. [Default: 0.11]
    :param y: y position of the button. [Default: 1.1]
    """
    order = ["log", "linear"] if primary_log else ["linear", "log"]

    # Check if the figure has a secondary y-axis
    has_secondary_y = "yaxis2" in fig.layout

    # Set args based on the presence of secondary y-axis
    if has_secondary_y:
        args_log = {"yaxis.type": order[0], "yaxis2.type": order[0]}
        args_linear = {"yaxis.type": order[1], "yaxis2.type": order[1]}
    else:
        args_log = {"yaxis.type": order[0]}
        args_linear = {"yaxis.type": order[1]}

    # Update layout to add buttons for switching scales
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                buttons=[
                    dict(
                        args=[args_log],
                        label=order[0],
                        method="relayout",
                    ),
                    dict(
                        args=[args_linear],
                        label=order[1],
                        method="relayout",
                    ),
                ],
                pad={"r": 10, "t": 10},
                showactive=True,
                x=x,
                xanchor="left",
                y=y,
                yanchor="top",
            ),
        ]
    )


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
        color=color,
    )
    fig.update_traces(marker=dict(size=3), selector=dict(mode="markers"))

    def make_button(eye, label, x_pos):
        button = dict(
            buttons=list(
                [
                    dict(
                        args=[
                            {
                                "scene.camera.eye": eye,
                                "scene.camera.up": {"x": 0, "y": 1, "z": 0},
                            }
                        ],
                        label=label,
                        method="relayout",
                    ),
                ]
            ),
            type="buttons",
            pad={"r": 10, "t": 10},
            showactive=True,
            x=x_pos,
            xanchor="right",
            y=1.1,
            yanchor="top",
        )
        return button

    def make_buttons(x, y, z):
        xy = make_button({"x": 0, "y": 0, "z": 2}, f"{x}-{y}", x_pos=0.6)
        zx = make_button({"x": 0, "y": 2, "z": 0}, f"{z}-{x}", x_pos=0.8)
        zy = make_button({"x": 2, "y": 0, "z": 0}, f"{z}-{y}", x_pos=1)
        return [xy, zx, zy]

    fig.update_layout(updatemenus=make_buttons(x, y, z))
    return fig


def heatmap_3D_plot(df, xaxis="x", params=None):
    """
    Plots params as 3D surface and heatmap for data exploration.
    :param df: Pandas DataFrame with MultiIndex Index that contains x and y values.
    :param xaxis: X axis name (string) or position (integer). [Default: 'x']
    :param params: List of params from Columns to plot.
    If None, plots all the Params. [Default: None]
    :returns: plotly.graph_objects.Figure instance.
    """
    if type(df) == pd.Series:
        params = [""]
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
                visible=visible,
                colorscale="Inferno",
            )
        )

    lpr = len(params)
    # if lpr == 1:
    # #     add_trace(fig, df)
    # else:
    bl = []
    for nr, param in enumerate(params):
        if len(params) == 1:
            add_trace(fig, df)
        elif nr == 0:
            add_trace(fig, df[param])
        else:
            add_trace(fig, df[param], visible=False)
        tfl = lpr * [False]
        tfl[nr] = True
        bl.append(dict(args=[{"visible": tfl}], label=param, method="update"))
    # fig.update_layout(
    #     updatemenus=[
    #          dict(
    #             buttons=bl,
    #             direction="down",
    #             pad={"r": 10, "t": 10},
    #             showactive=True,
    #             x=0.1,
    #             xanchor="left",
    #             y=1.1,
    #             yanchor="top"
    #         ),
    #     ])

    fig.update_layout(
        updatemenus=[
            dict(
                buttons=list(
                    [
                        dict(
                            args=["type", "heatmap"], label="Heatmap", method="restyle"
                        ),
                        dict(
                            args=["type", "surface"],
                            label="3D Surface",
                            method="restyle",
                        ),
                    ]
                ),
                # direction="down",
                type="buttons",
                direction="right",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.37,
                xanchor="left",
                y=1.1,
                yanchor="top",
            ),
            dict(
                buttons=bl,
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.1,
                xanchor="left",
                y=1.1,
                yanchor="top",
            ),
            dict(
                buttons=list(
                    [
                        dict(
                            args=[
                                {
                                    "scene.camera.eye": {"x": 0, "y": 0, "z": 2},
                                    "scene.camera.up": {"x": 0, "y": 1, "z": 0},
                                }
                            ],
                            label="Rotate back",
                            method="relayout",
                        ),
                    ]
                ),
                type="buttons",
                pad={"r": 10, "t": 10},
                x=1,
                xanchor="right",
                y=1.1,
                yanchor="top",
            ),
        ]
    )
    camera = dict(up=dict(x=0, y=1, z=0), eye=dict(x=0, y=0, z=2))
    fig.update_layout(
        margin=dict(t=100, b=0, l=0, r=0),
        scene_camera=camera,
        xaxis_title=xaxis,
        yaxis_title=df.index.name,
    )
    return fig


def add_multilayer_plot(
    fig: go.Figure,
    df: pd.DataFrame,
    plot_single,
    params=None,
    row=1,
    col=1,
    button_x=0.1,
    button_y=1.1,
    **kwargs,
):
    """_summary_

    :param fig: _description_
    :param df: _description_
    :param plot_single: _description_
    :param params: _description_, defaults to None
    :param row: _description_, defaults to 1
    :param col: _description_, defaults to 1
    :param button_x: _description_, defaults to 0.1
    :param button_y: _description_, defaults to 1.1
    """

    if params is None:
        params = df.columns

    # get the visibility of traces already on the figure
    append_is_visible = []
    for trace in fig.data:
        append_is_visible.append(trace.visible)

    def add_trace(fig, data, row, col, visible=True):
        traces = plot_single(data, visible=visible, **kwargs)
        if (row == 1) and (col == 1):
            fig.add_traces(traces)
        else:
            fig.add_traces(traces, rows=row, cols=col)
        return len(traces)

    param_len = len(params)
    bl = []
    for nr, param in enumerate(params):
        if nr == 0:
            trace_len = add_trace(fig, df[param], row, col)
        else:
            add_trace(fig, df[param], row, col, visible=False)
        is_visible = param_len * trace_len * [False]
        for i in range(trace_len):
            is_visible[nr * trace_len + i] = True
        is_visible = append_is_visible + is_visible
        bl.append(dict(args=[{"visible": is_visible}], label=param, method="update"))
    fig.update_layout(
        updatemenus=[
            dict(
                buttons=bl,
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=button_x,
                xanchor="left",
                y=button_y,
                yanchor="top",
            ),
        ]
    )


def multilayer_plot(df: pd.DataFrame, plot_single: go.Figure, params=None, **kwargs):
    """
    Takes
    :param df: Pandas DataFrame with columns as layers.
    :param plot_single: Function that returns object passed to Plotly's fig.add_trace() function.
    Takes one column of the dataframe (layer) and `visible` as arguments.
    :param params: List of params from Columns to plot.
    If None, plots all the Params. [Default: None]
    :param kwargs: Keyword arguments passed to plot_single function.
    :returns: plotly.graph_objects.Figure instance.
    """

    fig = go.Figure()
    add_multilayer_plot(fig, df, plot_single, params, **kwargs)

    return fig


def multilayer_image_plot(df, yaxis, cb_title=None):
    """
    Make multilayer plot to show images using bp.multilayer_plot.
    :param df: Pandas DataFrame.
    :param yaxis: Index level to use as y axis.
    :param cb_title: Color bar title. [Default: None]
    :returns: Plotly Figure.
    """

    def plot_single(data, visible, cb_title):
        plot_data = data.unstack(yaxis)
        plot_data = plot_data.dropna(axis=0, how="all")
        plot_data = plot_data.dropna(axis=1, how="all")
        trace = go.Heatmap(
            x=plot_data.index.values,
            y=plot_data.columns.values,
            z=plot_data,
            colorscale="Inferno",
            colorbar=dict(title=cb_title),
            transpose=True,
            visible=visible,
        )
        return [trace]

    multilayer_fig = multilayer_plot(df, plot_single, cb_title=cb_title)
    multilayer_fig.update_layout(
        updatemenus=[
            dict(
                x=1,
                xanchor="right",
                y=1.1,
                yanchor="top",
            ),
        ],
    )
    return multilayer_fig


# set the template


pio.templates["bix"] = go.layout.Template(
    layout={
        "annotationdefaults": {"arrowhead": 0, "arrowwidth": 1},
        "autotypenumbers": "strict",
        "coloraxis": {
            "colorbar": {
                "outlinewidth": 0,
                "tickcolor": "rgb(36,36,36)",
                "ticks": "outside",
            }
        },
        "colorscale": {
            "diverging": "balance",
            "sequential": "inferno",
            "sequentialminus": "bluered",
        },
        "colorway": [
            "#636EFA",
            "#EF553B",
            "#00CC96",
            "#AB63FA",
            "#FFA15A",
            "#19D3F3",
            "#FF6692",
            "#B6E880",
            "#FF97FF",
            "#FECB52",
        ],
        # 'toImageButtonOptions': {
        # 'format': 'svg',  # one of png, svg, jpeg, webp
        # 'filename': 'custom_image',
        # 'scale': 1  # Multiply title/legend/axis/canvas sizes by this factor
        # },
        "font": {
            "color": "rgb(0,0,0)",
            "family": "Helvetica,Futura,Calibri,Times New Roman",
            "size": 18,
        },
        "geo": {
            "bgcolor": "white",
            "lakecolor": "white",
            "landcolor": "white",
            "showlakes": True,
            "showland": True,
            "subunitcolor": "white",
        },
        "hoverlabel": {"align": "left"},
        "hovermode": "closest",
        "legend": {
            "bgcolor": "rgba(255,255,255,0.7)",
            "xanchor": "left",
            "yanchor": "top",
            "x": 0.01,
            "y": 0.99,
        },
        "mapbox": {"style": "light"},
        "paper_bgcolor": "white",
        "plot_bgcolor": "white",
        "polar": {
            "angularaxis": {
                "gridcolor": "rgb(232,232,232)",
                "linecolor": "rgb(36,36,36)",
                "showgrid": False,
                "showline": True,
                "ticks": "outside",
            },
            "bgcolor": "white",
            "radialaxis": {
                "gridcolor": "rgb(232,232,232)",
                "linecolor": "rgb(36,36,36)",
                "showgrid": False,
                "showline": True,
                "ticks": "outside",
            },
        },
        "scene": {
            "xaxis": {
                "backgroundcolor": "white",
                "gridcolor": "rgb(232,232,232)",
                "gridwidth": 2,
                "linecolor": "rgb(36,36,36)",
                "showbackground": True,
                "showgrid": False,
                "showline": True,
                "ticks": "outside",
                "zeroline": False,
                "zerolinecolor": "rgb(36,36,36)",
            },
            "yaxis": {
                "backgroundcolor": "white",
                "gridcolor": "rgb(232,232,232)",
                "gridwidth": 2,
                "linecolor": "rgb(36,36,36)",
                "showbackground": True,
                "showgrid": False,
                "showline": True,
                "ticks": "outside",
                "zeroline": False,
                "zerolinecolor": "rgb(36,36,36)",
            },
            "zaxis": {
                "backgroundcolor": "white",
                "gridcolor": "rgb(232,232,232)",
                "gridwidth": 2,
                "linecolor": "rgb(36,36,36)",
                "showbackground": True,
                "showgrid": False,
                "showline": True,
                "ticks": "outside",
                "zeroline": False,
                "zerolinecolor": "rgb(36,36,36)",
            },
        },
        "shapedefaults": {"fillcolor": "black", "line": {"width": 0}, "opacity": 0.3},
        "ternary": {
            "aaxis": {
                "gridcolor": "rgb(232,232,232)",
                "linecolor": "rgb(36,36,36)",
                "showgrid": False,
                "showline": True,
                "ticks": "outside",
            },
            "baxis": {
                "gridcolor": "rgb(232,232,232)",
                "linecolor": "rgb(36,36,36)",
                "showgrid": False,
                "showline": True,
                "ticks": "outside",
            },
            "bgcolor": "white",
            "caxis": {
                "gridcolor": "rgb(232,232,232)",
                "linecolor": "rgb(36,36,36)",
                "showgrid": False,
                "showline": True,
                "ticks": "outside",
            },
        },
        "title": {"x": 0.5, "y": 0.9, "xanchor": "center", "yanchor": "top"},
        "xaxis": {
            "automargin": True,
            "gridcolor": "rgb(232,232,232)",
            "linecolor": "rgb(36,36,36)",
            "showgrid": False,
            "showline": True,
            "ticks": "outside",
            "title": {"standoff": 15},
            "zeroline": False,
            "zerolinecolor": "rgb(170,170,170)",
            "mirror": True,
        },
        "yaxis": {
            "automargin": True,
            "gridcolor": "rgb(232,232,232)",
            "linecolor": "rgb(36,36,36)",
            "showgrid": False,
            "showline": True,
            "ticks": "outside",
            "title": {"standoff": 15},
            "zeroline": False,
            "zerolinecolor": "rgb(170,170,170)",
            "mirror": True,
        },
    }
)

pio.templates.default = "bix"
