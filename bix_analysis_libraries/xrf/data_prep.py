from molmass import Formula


def calc_molar(df):
    '''
    :param df: DataFrame with element names as columns.
    '''
    mol_df = df.copy()
    for name in df:
        mol_df[name] = df[name] / Formula(name).mass
