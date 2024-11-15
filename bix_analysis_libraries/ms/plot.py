from plotly import graph_objects as go


def plot_ms(
    fig, df, row=1, col=1, legend1="legend1", legend2="legend2", show_only_first=False
):
    for column in df.columns:
        visible = True
        if show_only_first:
            visible = True if column == df.columns[0] else "legendonly"
        secondary_y = True if column[0] == "SEM" else False
        legend = legend2 if secondary_y else legend1
        data = df[column].values
        fig.add_trace(
            go.Scatter(
                x=df.index.values,
                y=data,
                name=column[1],
                legend=legend,
                visible=visible,
                mode="lines+markers",
            ),
            secondary_y=secondary_y,
            row=row,
            col=col,
        )
