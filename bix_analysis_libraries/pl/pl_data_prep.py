import scipy.constants as phys
from bric_analysis_libraries.pl import pl_data_prep as pldp


def calc_photon_flux(df, calib, area):
    '''
    Calculates photon flux [photons/(m2 s nm)] from counts [1/(s nm)].
    :param df: DataFrame indexed by wavelength
    :param calib: DataFrame w/ calibration and dark offset curves.  
    :param area: Spot size in m2
    :returns: Dataframe with calculated 
    '''
    def calc(x):
        return (x - calib.offset) * calib.calib / area

    # remove first values if the legths don't coincide
    # len_diff = len(df) - len(offset)
    # if len_diff > 0:
    #     df = df.iloc[len_diff:]

    return df.apply(calc)


def jacobian_wl_to_eng(df):
    '''
    Applies Jacobian transformation to columns in a dataframe indexed by energy.
    param df: DataFrame indexed by energy.
    returns: DataFrame with transformed columns
    '''
    const = 1e9 * phys.Planck * phys.c / phys.e
    return df.apply(lambda x: const * x / (x.index.values ** 2))


def df_to_energy(df):
    '''
    Converts index from wavelength to energy and applies Jacobian transformation to the columns.
    param df: DataFrame indexed by wavelength
    returns: DataFrame indexed by energy with transformed columns
    '''
    e_df = pldp.index_to_energy(df)
    e_df = jacobian_wl_to_eng(e_df)
    e_df.index = e_df.index.rename('energy')
    return e_df.sort_index()
