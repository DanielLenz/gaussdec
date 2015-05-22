import tables
import numpy as np
import healpy as hp

def make_average_linewidth_map(table):

    nside = 1024
    npix = hp.nside2npix(nside)
    linewidths = np.zeros(npix, dtype=np.float32)
    amplitudes = np.zeros(npix, dtype=np.float32)

    for i, row in enumerate(table.iterrows()):
        if not i % 100000:
            print i
        linewidths[row['hpxindex']] += row['sigma_kms'] * row['amplitude']
        amplitudes[row['hpxindex']] += row['amplitude']
        
    weighted_linewidths = linewidths/amplitudes
    weighted_linewidths[weighted_linewidths==0] = np.nan
    
    return weighted_linewidths

def reconstruct_coldens(table, sigmas=[0., np.inf]):
    """
    Reconstruct a column density map of the full sky at nside=1024
    """
    npix = hp.nside2npix(1024)
    hi_model = np.zeros(npix, dtype=np.float32)

    for row in table:
        if (sigmas[0] < row['sigma_kms'] < sigmas[1]):
            hi_model[row['hpxindex']] += row['amplitude']

    # convert to cm**-2, 1.288 is EBHIS chanwidth
    to_coldens = 1.82e18 * 1.288

    return hi_model * to_coldens



if __name__ == '__main__':
    # load gaussdec
    # store = tables.open_file('/vol/ebhis1/data1/dlenz/projects/gaussdec/data/camargo_gdec.h5', mode='r')
    store = tables.open_file('/users/dlenz/projects/carmago/data/carmago_gdec.h5', mode='r')
    gaussdec = store.root.gaussdec

    weighted_linewidths = make_average_linewidth_map(gaussdec)
    np.save('data/weighted_linewidths.npy', weighted_linewidths)

    cold_components = reconstruct_coldens(gaussdec, sigmas=[0., 4.25])
    np.save('data/cold_components.npy', cold_components)