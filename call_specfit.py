from multiprocessing import Pool
import argparse
import os
import itertools as it
import glob
import tables
import numpy as np
import warnings

import healpy as hp
from astropy.io import fits

from specfitting import fit_spectrum, make_multi_gaussian_model, default_p


def fit_file(args):

    filename, p = args

    print filename, p
    # create theano functions
    f_model, f_residual, f_objective, f_jacobian, f_stats = make_multi_gaussian_model()

    # read file
    hpxfile = fits.getdata(filename, ext=1)

    def fit_spectra():
        for i, row in enumerate(hpxfile):
            # print '{n} of {tot}'.format(n=i, tot=len(hpxfile))

            theta, glon = np.rad2deg(hp.pix2ang(1024, row['HPXINDEX']))
            glat = 90. - theta
            yield int(row['HPXINDEX']), fit_spectrum(row['DATA'], f_objective, f_jacobian, f_stats, p)

    # put results into dict, dump them to disk
    results = {int(k): v for k, v in fit_spectra()}

    return results


def gen_file_fit(args):
    # get filenames
    filenames = glob.glob(args.glob)

    # set default parameters
    p = default_p
    p['clobber'] = False
    p['survey'] = args.survey

    for filename, p in it.izip(filenames, it.repeat(p)):
        yield filename, p


def create_tables(args):

    # append to an existing h5file
    if (os.path.exists(args.outname)):
        warnings.warn('File {} already exists, will add entries.'.format(args.outname))

        return

    # create a new h5file and append entries
    else:
        print 'Creating file {} ...'.format(args.outname)

        class GaussDec(tables.IsDescription):
            # coordinates
            index = tables.Int32Col()
            glon = tables.Float32Col()
            glat = tables.Float32Col()

            # Gauss fit parameters
            amplitude = tables.Float32Col()
            
            center_c = tables.Float32Col()
            center_kms = tables.Float32Col()
            
            width_c = tables.Float32Col()
            width_kms = tables.Float32Col()
    
    with tables.open_file(args.outname, mode="w", title="Gaussdec") as fobj:
            fobj.create_table(fobj.root, 'gaussdec_ebhis', GaussDec, "Gauss decomposition EBHIS")
            fobj.create_table(fobj.root, 'gaussdec_gass', GaussDec, "Gauss decomposition GASS")

    return 0


def fit_files(args):
    # create a pool, fit all files
    with tables.open_file(args.outname, mode="a", title="Gaussdec") as fobj:
        if args.survey == 'EBHIS':
            table = fobj.root.gaussdec_ebhis

        if args.survey == 'GASS':
            table = fobj.root.gaussdec_gass

        p = Pool()
        for fitresults in p.imap(fit_file, gen_file_fit(args)):
            for k, v in fitresults.iteritems():
                theta, glon = np.rad2deg(hp.pix2ang(1024, int(k)))
                glat = 90. - theta
                for i in range(len(v['parameters'])/3):
                    entry = table.row
                    entry['index'] = int(k)
                    entry['glon'] = glon
                    entry['glat'] = glat

                    entry['amplitude'] = v['parameters'][i*3]
                    entry['center_c'] = v['parameters'][i*3 + 1]
                    entry['center_kms'] = np.nan

                    entry['width_c'] = v['parameters'][i*3 + 2]
                    entry['width_kms'] =  np.nan

                    entry.append()
        table.cols.index.create_csindex()
        table.flush()

    return 0

# main
if __name__ == '__main__':
    
    # evaluate parsed arguments
    argp = argparse.ArgumentParser(description='Generate a Gauss decomposition of the full EBHIS/GASS sky.')
    argp.add_argument('-s', '--survey',
                default='EBHIS',
                metavar='survey',
                choices = ['GASS', 'EBHIS'],
                help='Survey that is used for the decomposition.',
                type=str)
    argp.add_argument('-g', '--glob',
                default='/vol/arc1/archive/bwinkel/EBHIShpx/ebhis_hpx_1024*of192_G7.fits',
                metavar='glob pattern',
                help='Glob pattern to find the source hpx-files.',
                type=str)
    argp.add_argument('outname',
                metavar='output_filename',
                type=str)
    argp.add_argument('-c', '--clobber',
                default=False,
                metavar='clobber',
                help='clobber',
                type=bool)

    args = argp.parse_args()

    # check and create output h5file
    create_tables(args)
    
    # fit files
    fit_files(args)
















