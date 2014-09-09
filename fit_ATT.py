import json
import gzip
import os
import itertools as it

import numpy as np

from astropy.io import fits

from fitting import fit_spectrum, make_multi_gaussian_model, default_p


def fit_file(filename, p):

    outname = os.path.join(os.path.basename(filename), '_fits.gzjs')

    if not os.path.exists(outname):

        # create thena functions
        f_model, f_residual, f_objective, f_jacobian, f_stats = fitting.make_multi_gaussian_model()

        # read file
        hpxfile = fits.getdata(filename, ext=1)

        def fit_spectra():
            for y in hpxfile['DATA']:
                yield fit_spectrum(y, f_objective, f_jacobian, f_stats, p)

        results = {k : v for k, v in it.izip(hpxfile['HPXINDEX'], fit_spectra())}

        with gzip.GzipFile(outname, 'w') as f:
            json.dump(results, f)


def gen_file_fit():
    # get filenames
    filenames = glob.glob('/vol/arc1/archive/bwinkel/EBHIShpx/ebhis_hpx_1024*141of192_G.fits')
    
    # set default parameters
    p = default_p

    for f, p in it.izip(filenames, it.repeat(p))
        yield (f, p)

# main
if __name__ == '__main__':

    # fit a single file
    apply(fit_file, gen_file_fit)






