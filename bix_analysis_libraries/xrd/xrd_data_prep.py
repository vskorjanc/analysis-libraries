from BaselineRemoval import BaselineRemoval
import pandas as pd


def subtract_background(df):
    sub_dfs = {}
    for column in df:
        df_wo_na = df[column].dropna()
        baseObj = BaselineRemoval(df_wo_na)
        subtracted = baseObj.ZhangFit()
        sub_df = pd.Series(subtracted, index=df_wo_na.index)
        sub_dfs[column] = sub_df
    sub_dfs = pd.concat(sub_dfs, axis=1, names=df.columns.names)
    return sub_dfs
