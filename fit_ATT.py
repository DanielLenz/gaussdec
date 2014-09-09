from multiprocessing import Pool
import json
import gzip
import os
import itertools as it
import glob
import numpy as np

import healpy as hp
from astropy.io import fits

from fitting import fit_spectrum, make_multi_gaussian_model, default_p


def fit_file(args):

    filename, p = args
    outname = os.path.splitext(filename)[0] + '_fits.gzjs'

    if not os.path.exists(outname):

        # create theano functions
        f_model, f_residual, f_objective, f_jacobian, f_stats = make_multi_gaussian_model()

        # read file
        hpxfile = fits.getdata(filename, ext=1)

        def fit_spectra():
            for row in hpxfile[:50]:
                # skip the Galactic plane
                theta, glon = np.rad2deg(hp.pix2ang(1024, row['HPXINDEX']))
                glat = 90. - theta
                if glat > 15.:
                    yield row['HPXINDEX'], fit_spectrum(row['DATA'], f_objective, f_jacobian, f_stats, p)

        # put results into dict, dump them to disk
        results = {int(k) : v for k, v in fit_spectra()}

        with gzip.GzipFile(outname, 'w') as f:
            json.dump(results, f)


def gen_file_fit():
    # get filenames
    filenames = glob.glob('/users/dlenz/projects/gaussdec/ebhis_hpx_1024*112of192_G7.fits')
    
    # set default parameters
    p = default_p

    for f, p in it.izip(filenames, it.repeat(p)):
        yield f, p

# main
if __name__ == '__main__':

    # fit a single file
    # p = Pool()
    # p.apply(fit_file, gen_file_fit)
    map(fit_file, gen_file_fit())







