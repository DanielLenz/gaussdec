import bottleneck as bn
import json
import gzip
import argparse
import itertools as it
import numpy as np
import glob
from functools import partial
from scipy import stats
from multiprocessing import Pool
import cPickle

import healpy as hp

def get_mask(indices, nside):
    npix = hp.nside2npix(nside)
    
    mask = np.zeros(npix, dtype=bool)
    for index in hpxindices:
        mask[index] = True

    return mask

def get_hi_model(fitresults, func, npix=hp.nside2npix(2**10)):
    """
    Reconstruct the HI column density, depending on the line width of the components.

    Input
    -----
    fitresults : dict
        For each hpx-pixel, contains a parameter keyword with the fit results.

    func : function(amp, x0, sigma)
        Function based on a, x, s that determines whether the component is added to the model or not.

    Return
    ------
    hi_model : ndarray 

    """
    hi_model = np.zeros(npix, dtype=np.float32) * np.nan

    for k, v in fitresults.iteritems():
        p = np.array(v['parameters'])

        hi_model[int(k)] = sum(it.compress(
                                p[::3],
                                [func(a, x, s) for a, x, s in p.reshape(p.size/3, 3)]
                                ))
    return hi_model

def get_corr_coeffs(fitresults, dust, npix, mask, s_thresh):
    hi_model = get_hi_model(fitresults, s_thresh)

    rho, p = stats.spearmanr(a=hi_model[mask], b=dust[mask], axis=None)

    return rho

def maximum_correlation(fitresults, source, npix=hp.nside2npix(2**10)):
    
    # load dust data for correlation
    tau_full = hp.read_map(source + 'opacity_10.83.hpx.fits', dtype=np.float32)
    tau = hp.ud_grade(tau_full, nside)

    # create mask for fitted hpxindices
    hpxindices = np.array(fitresults.keys(), dtype=int)
    mask = get_mask(hpxindices, nside=nside)

    # create theano functions
    f_model, f_residual, f_objective, f_jacobian, f_stats = make_multi_gaussian_model()
    
    get_corr_coeffs_par = partial(get_corr_coeffs, fitresults, tau, npix, mask)
    
    # determine correlation coefficients, depending on the line-widths used
    s_threshs = np.linspace(1.5, 6., 15.)
    corr_coeffs = map(get_corr_coeffs_par, s_threshs)

    # return (cold_comp, warm_comp)

def fixed(fitresults):
    """
    Use a fixed threshold in line width to reconstruct
    cold and warm components
    """

    # convert to cm**-2, 1.288 is EBHIS chanwidth
    to_coldens = 1.82e18 * 1.288

    components = {}
    components['warm'] = get_hi_model(fitresults, lambda a, x, s: s > 2.) * to_coldens
    components['cold'] = get_hi_model(fitresults, lambda a, x, s: s < 2.) * to_coldens

    return components

def reconstruct(source, method='fixed'):
    fitresults = cPickle.load(open(source + 'fitresults.b', 'rb'))
    
    if method == 'fixed':
        return fixed(fitresults)

    if method == 'maxcorr':
        return maximum_correlation(fitresults, source)


def gzjs2pickle(source):
    """
    Not in active use, converts the numerous json files to a single pickle
    """

    # load fitresults from json zip files
    result_filenames = glob.glob(source + '*.gzjs')

    fitresults = {}
    for filename in result_filenames:
        fitresults.update(json.load(gzip.GzipFile(filename)))

    cPickle.dump(fitresults, open('fitresults.b', 'wb'), protocol=2)

if __name__ == '__main__':
    pass


