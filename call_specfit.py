"""
Generate a Gaussian decomposition of spectra, based on and written to hd5-files
"""

"""
Functions
---------
create_tables(args) : If the table exists, abort. Else, create a new hdf5-table
    where the decomposition is saved.
initializer(infile) : Prepare the Gaussian model and the input file, needed for
    multiprocessing
do_fit(row_index) : Fit a given row of the input file
get_row_index(nsamples, table) : Yield all the rows of the input file or a
    randomly chosen sample
fit_spectra(args) : Read the input file, create the pool,
    assign the fitting jobs and write the results to disk
"""

import warnings
from multiprocessing import Pool
import argparse
import os
import healpy as hp
import tables
import numpy as np
from datetime import datetime

from specfitting import fit_spectrum, make_multi_gaussian_model, default_p

# store = tables.open_file('../survey2pytable/data/bps.h5')
# f_model, f_residual, f_objective, f_jacobian, f_stats = make_multi_gaussian_model()

# ebhis standard spectral restframe
CRPIX3 = 471.921630003202
CDELT3 = 1288.23448620083


class GaussDec(tables.IsDescription):
    """
    Description for the pytable, specifying the columns
    and their data types
    """
    # coordinates
    hpxindex = tables.Int32Col()
    glon = tables.Float32Col()
    glat = tables.Float32Col()

    # Gauss fit parameters
    amplitude = tables.Float32Col()

    center_c = tables.Float32Col()
    center_kms = tables.Float32Col()

    sigma_c = tables.Float32Col()
    sigma_kms = tables.Float32Col()


def chan2velo(channel):
    """
    Convert Channel to LSR velocity in m/s
    """
    return (channel - CRPIX3) * CDELT3


def create_tables(arguments):
    """
    If the table exists, abort. Else, create a new hdf5-table where the
    decomposition is saved.
    """

    # Read or create file
    if os.path.isfile(arguments.outname) and not arguments.clobber:
        warnings.warn('File {} already exists. Appending...'.format(
            arguments.outname))
        return
    else:
        print "Creating file {}".format(arguments.outname)

    store = tables.open_file(arguments.outname, mode="w")

    # check for existing tables
    gaussdec = store.create_table(
        store.root,
        'gaussdec',
        GaussDec,
        "Gauss decomposition")
    gaussdec.cols.hpxindex.create_csindex()
    gaussdec.autoindex = True

    return 0


def initializer(infile):
    """
    Prepare the Gaussian model and the input file, needed for
    multiprocessing
    """

    global f_model, f_residual, f_objective, f_jacobian, f_stats

    # create theano functions
    f_model, f_residual, f_objective, f_jacobian, f_stats = make_multi_gaussian_model()

    global store
    store = tables.open_file(infile)

    return None


def do_fit(row_index):
    """
    Fit a given row of the input file
    """
    table = store.root.survey

    row = table[row_index]
    fitresults = fit_spectrum(
        row,
        f_objective,
        f_jacobian,
        f_stats,
        default_p)['parameters']

    return row_index, fitresults


def get_row_index(nsamples, hpxindices, table):
    """
    Yield all the rows of the input file or a randomly chosen sample
    """
    if hpxindices is None:

        if nsamples < 0:
            for row_index in range(table.nrows):
                if not row_index % 10000:
                    print 'Working on row {i} of {n}...'.format(
                        i=row_index,
                        n=table.nrows)
                yield row_index
        else:
            sample_indices = np.random.choice(
                range(table.nrows),
                size=nsamples,
                replace=False)

            for i, row_index in enumerate(sample_indices):
                if not i % 1000:
                    print 'Working on row {i} of {n}..'.format(i=i, n=nsamples)
                yield row_index
    else:
        indices = np.load(hpxindices)
        for i, row_index in enumerate(indices):
            if not i % 1000:
                print 'Working on row {i} of {n}..'.format(i=i, n=len(indices))
            yield row_index


def fit_spectra(arguments):
    """
    Read the input file, create the pool, assign the fitting jobs and write
    the results to disk
    """
    # create a pool, fit all files
    with tables.open_file(arguments.outname, mode="a") as gdec_store:

        pool = Pool(
            initializer=initializer,
            initargs=(arguments.infile,))

        infile_store = tables.open_file(arguments.infile)
        infile_table = infile_store.root.survey

        gdec_table = gdec_store.root.gaussdec

        for row_index, fitresults in pool.imap(
            do_fit,
            get_row_index(
                arguments.nsamples,
                arguments.hpxindices,
                infile_table)):
            hpxindex = row_index
            theta, glon = np.rad2deg(hp.pix2ang(1024, hpxindex))
            glat = 90. - theta

            for i in range(len(fitresults)/3):
                entry = gdec_table.row
                entry['hpxindex'] = hpxindex
                entry['glon'] = glon
                entry['glat'] = glat

                entry['amplitude'] = fitresults[i * 3]
                entry['center_c'] = fitresults[i * 3 + 1]
                entry['center_kms'] = chan2velo(entry['center_c']) / 1.e3

                entry['sigma_c'] = fitresults[i * 3 + 2]
                entry['sigma_kms'] = entry['sigma_c'] * CDELT3 / 1.e3

                entry.append()

            if row_index % 1000 == 0:
                gdec_store.flush()

        infile_store.close()
        gdec_store.close()

    return 0


def main():
    """
    Create the argparser, create the tables and perform the fit
    """

    # evaluate parsed arguments
    argp = argparse.ArgumentParser(description=__doc__)

    argp.add_argument(
        '-i',
        '--infile',
        default='/vol/ebhis2local/data1/dlenz/projects/survey2pytable/data/bps.h5',
        metavar='infile',
        help='Source pytable',
        type=str)

    argp.add_argument(
        '-n',
        '--nsamples',
        default=-1,
        metavar='nsamples',
        help='Number of random sightlines that are fitted',
        type=int)

    argp.add_argument(
        '-x',
        '--hpxindices',
        default=None,
        metavar='hpxindices',
        help='Location of a npy file that contains the hpx indices to fit',
        type=str)

    argp.add_argument(
        '-c',
        '--clobber',
        default=False,
        metavar='clobber',
        help='clobber',
        type=bool)

    argp.add_argument(
        'outname',
        metavar='output_filename',
        type=str)

    args = argp.parse_args()

    # check and create output h5file
    create_tables(args)

    # fit files
    fit_spectra(args)


# main
if __name__ == '__main__':
    tstart = datetime.now()
    main()
    print 'Runtime: {}'.format(datetime.now() - tstart)

























