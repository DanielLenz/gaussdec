import numpy as np

from astropy.io import fits

from fitting import fit_spectrum, make_multi_gaussian_model


def fit_file(filename, p):
    # create thena functions
    f_model, f_residual, f_objective, f_jacobian, f_stats = fitting.make_multi_gaussian_model()

    # read file
    hpxfile = fits.getdata(filename, ext=1)
    values = hpxfile['DATA']

    # fit spectrum
    fit_results = fitting.fit_spectrum(values, f_objective, f_jacobian, f_stats, p)
    
    # add entry to dict
    dict_entry = {}
    dict_entry['hpxindex'] = hpxfile['HPXINDEX']
    dict_entry['fit_results'] = fit_results

    return dict_entry

def gen_file_fit():
    # get filenames
    filenames = glob.glob('/vol/arc1/archive/bwinkel/EBHIShpx/ebhis_hpx_1024*141of192_G.fits')
    
    # set default parameters
    p = {
    'min_components' : 1,
    'max_components' : 10,
    'iterations' : 10,
    'int_low' : 5e18 / 1.82e18 / 1.28,
    'int_high' : 1e21 / 1.82e18 / 1.28,
    'sigma_low' : np.sqrt(50 / 21.85) / 1.28 / 2.35,
    'sigma_high' : np.sqrt(10000 / 21.85) / 1.28 / 2.35,
    'pdf_threshold' : 0.1,
    'pdf_kernel' : 3.32, 
    'fit_method' : 'l-bfgs-b',
    }

    for f, p in it.izip(filenames, it.repeat(p))
        yield (f, p)

# main
if __name__ == '__main__':
    
    results = {}

    # fit a single file
    dict_entry = map(fit_file, gen_file_fit)

    # add corresponding entry to results dict
    results[dict_entry['hpxindex']] = dict_entry['fit_results']







