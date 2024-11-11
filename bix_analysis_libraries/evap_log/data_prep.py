import pandas as pd


def prep_shut(shut):
    """Apply pre-analysis data handling to shutter posision data.

    :param shut: Pandas series ('Shutter_Substrat' row)
    :returns: Pandas series
    """
    return shut.rolling(5, min_periods=1, center=True).median()


def filter_scale_pressure(pres: pd.Series) -> pd.Series:
    """Filter and scale pressure values for plotting.

    :param pres: Pandas series ('P_Chamber row').
    :returns: Pandas series.
    """
    pres_filter = (
        pres.rolling(15, center=True, min_periods=1)
        .median()
        .rolling(15, center=True, min_periods=1)
        .mean()
    )
    pres_filter = (
        pres_filter.diff().rolling(30, center=True, min_periods=1).mean().abs() < 3e-7
    )
    filtered_pres = pres.where(pres_filter)
    filtered_pres = filtered_pres.where(lambda x: x < 0.0001)
    filtered_pres = filtered_pres.apply(lambda x: x * 1e6)
    return filtered_pres
