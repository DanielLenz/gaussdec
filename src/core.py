import healpy as hp
import numpy as np

from myhelpers.datasets import hi4pi
from myhelpers import misc

import site

site.addsitedir(misc.bpjoin("gaussdec"))

from src import this_project as P


def reconstruct_coldens(table, full=False):
    """
    Reconstruct a column density map of the full sky at nside=1024
    If full, reconstruct not only NHI_tot, but also LVC/IVC/HVC (negative and positive velocities)
    """
    npix = hp.nside2npix(1024)
    nhi_tot = np.zeros(P.NPIX, dtype=np.float32)
    if full:
        nhi_lvc_neg = np.zeros(P.NPIX, dtype=np.float32)
        nhi_ivc_neg = np.zeros(P.NPIX, dtype=np.float32)
        nhi_hvc_neg = np.zeros(P.NPIX, dtype=np.float32)
        nhi_lvc_pos = np.zeros(P.NPIX, dtype=np.float32)
        nhi_ivc_pos = np.zeros(P.NPIX, dtype=np.float32)
        nhi_hvc_pos = np.zeros(P.NPIX, dtype=np.float32)

    for row in table:
        nhi_tot[row["hpxindex"]] += row["amplitude"]
        if full:
            v_lsr = row["center_kms"]
            if -650 < v_lsr < -90:
                nhi_hvc_neg[row["hpxindex"]] += row["amplitude"]
            if -90 < v_lsr < -45:
                nhi_ivc_neg[row["hpxindex"]] += row["amplitude"]
            if -45 < v_lsr < 0:
                nhi_lvc_neg[row["hpxindex"]] += row["amplitude"]
            if 0 < v_lsr < 45:
                nhi_lvc_pos[row["hpxindex"]] += row["amplitude"]
            if 45 < v_lsr < 90:
                nhi_ivc_pos[row["hpxindex"]] += row["amplitude"]
            if 90 < v_lsr < 650:
                nhi_hvc_pos[row["hpxindex"]] += row["amplitude"]

    # Lastly, convert from K.km/s to NHI
    if full:
        return dict(
                tot=nhi_tot * hi4pi.cK2nhi,
                lvc_neg=nhi_lvc_neg * hi4pi.cK2nhi,
                ivc_neg=nhi_ivc_neg * hi4pi.cK2nhi,
                hvc_neg=nhi_hvc_neg * hi4pi.cK2nhi,
                lvc_pos=nhi_lvc_pos * hi4pi.cK2nhi,
                ivc_pos=nhi_ivc_pos * hi4pi.cK2nhi,
                hvc_pos=nhi_hvc_pos * hi4pi.cK2nhi,
                )
    else:
        return nhi_tot * hi4pi.cK2nhi


def make_ncomp_map(table):
    """
    Create a map of the number of components, used to model the HI emission
    """

    ncomps = np.zeros(P.NPIX, dtype=np.int)

    for row in table:
        ncomps[row["hpxindex"]] += 1

    return ncomps
