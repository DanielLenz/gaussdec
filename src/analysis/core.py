import healpy as hp
import numpy as np

from . import this_project as P


def reconstruct_coldens(table):
    """
    Reconstruct a column density map of the full sky at nside=1024
    """
    npix = hp.nside2npix(1024)
    hi_model = np.zeros(P.NPIX, dtype=np.float32)

    for row in table:
        hi_model[row['hpxindex']] += row['amplitude']

    # convert to cm**-2, 1.288 is EBHIS chanwidth
    to_coldens = 1.82e18 * 1.288

    return hi_model * to_coldens


def make_ncomp_map(table):
    """
    Create a map of the number of components, used to model the HI emission
    """

    ncomps = np.zeros(P.NPIX, dtype=np.int)

    for row in table:
        ncomps[row['hpxindex']] += 1

    return ncomps
