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

from mathlib import mathfuncs as mf

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


def continuous(fitresults, x0=2., tau=0.3):

    def get_model_logi(fitresults, x0, tau, npix=hp.nside2npix(2**10)):
        hi_cold = np.zeros(npix, dtype=np.float32)
        hi_warm = np.zeros(npix, dtype=np.float32)

        for k, v in fitresults.iteritems():
            p = np.array(v['parameters'])
            for comp in p.reshape(p.size/3, 3):
                hi_cold[int(k)] += comp[0] * mf.logistic(comp[2], x0, -tau)
                hi_warm[int(k)] += comp[0] * mf.logistic(comp[2], x0, tau)

        return hi_cold, hi_warm

    # convert to cm**-2, 1.288 is EBHIS chanwidth
    to_coldens = 1.82e18 * 1.288

    components = {}
    components['cold'], components['warm'] = get_model_logi(fitresults, x0=2., tau=0.3)
    components['cold'] *= to_coldens
    components['warm'] *= to_coldens

    return components


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


def reconstruct(source, method='fixed', min_b=30.):
    fitresults = cPickle.load(open(source + 'fitresults.b', 'rb'))

    hpxindices = fitresults.keys()
    for k in hpxindices:
        theta, glon = np.rad2deg(hp.pix2ang(1024, int(k), nest=False))
        glat = 90. - theta
        if np.abs(glat) < min_b:
            fitresults.pop(k)
    
    if method == 'fixed':
        return fixed(fitresults)

    if method == 'continuous':
        return continuous(fitresults, tau=0.3)


def gzjs2pickle(source):
    """
    Not in active use, converts the numerous json files to a single pickle
    """

    # load fitresults from json zip files
    result_filenames = glob.glob(source + '*.gzjs')

    fitresults = {}
    for filename in result_filenames:
        fitresults.update(json.load(gzip.GzipFile(filename)))

    cPickle.dump(fitresults, open(source + 'fitresults.b', 'wb'), protocol=2)

if __name__ == '__main__':
    pass


