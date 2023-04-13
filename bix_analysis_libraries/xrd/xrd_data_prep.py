from BaselineRemoval import BaselineRemoval
import pandas as pd


def subtract_background(df):
    sub_df = pd.DataFrame(index=df.index, columns=df.columns)
    for column in df:
        baseObj = BaselineRemoval(df[column].dropna())
        subtracted = baseObj.ZhangFit()
        sub_df[column] = subtracted
    return sub_df
