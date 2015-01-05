"""
Generate a Gaussian decomposition of spectra, based on and written to hd5-files
"""

"""
Functions
---------
create_tables(args) : If table exists, abort. Else, create a new hdf5-table
    where the decomposition is saved.
initializer(infile) : Prepare the Gaussian model and the input file, needed for
    multiprocessing
do_fit(row_index) : Fit a given row of the input file
get_row_index(nsamples, table) : Yield all the rows of the input file or a
    randomly chosen sample
fit_spectra(args) : Reads the input file, creates the pool,
    assigns the fitting jobs and write to results to disk
"""

from multiprocessing import Pool
import argparse
import os
import tables
import numpy as np
import warnings

from specfitting import fit_spectrum, make_multi_gaussian_model, default_p

# ebhis standard spectral restframe
CRPIX3 = 471.921630003202
CDELT3 = 1288.23448620083

def chan2velo(channel):
    """
    Convert Channel to LSR velocity
    """
    return (channel - CRPIX3) * CDELT3


def create_tables(arguments):
    """
    If table exists, abort. Else, create a new hdf5-table where the
    decomposition is saved.
    """

    # append to an existing h5file
    if os.path.exists(arguments.outname):
        warnings.warn('File {} already exists!'.format(arguments.outname))
        return

    # create a new h5file and append entries
    else:
        print 'Creating file {} ...'.format(arguments.outname)

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

            width_c = tables.Float32Col()
            width_kms = tables.Float32Col()

    with tables.open_file(arguments.outname, mode="w", title="Gaussdec") as store:
        ebhis = store.create_table(
            store.root,
            'gaussdec_ebhis',
            GaussDec,
            "Gauss decomposition EBHIS")
        ebhis.cols.hpxindex.create_csindex()
        ebhis.autoindex = True

        # gass = fobj.create_table(fobj.root,
                                # 'gaussdec_gass',
                                # GaussDec,
                                # "Gauss decomposition GASS")
        # gass.cols.hpxindex.create_csindex()

    return 0


def initializer(infile):
    """
    Prepare the Gaussian model and the input file, needed for
    multiprocessing
    """

    global f_model, f_residual, f_objective, f_jacobian, f_stats

    # create theano functions
    f_model, f_residual, f_objective, f_jacobian, f_stats = make_multi_gaussian_model()

    global ebhis_file
    ebhis_file = tables.open_file(infile, mode="r", title="EBHIS")


def do_fit(row_index):
    """
    Fit a given row of the input file
    """
    row = ebhis_file.root.ebhis[row_index]
    fitresults = fit_spectrum(
        row['DATA'],
        f_objective,
        f_jacobian,
        f_stats,
        default_p)['parameters']

    return row_index, fitresults


def get_row_index(nsamples, table):
    """
    Yield all the rows of the input file or a randomly chosen sample
    """
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
                print 'Working on row {i} of {n}...'.format(i=i, n=nsamples)
            yield row_index


def fit_spectra(arguments):
    """
    Reads the input file, creates the pool, assigns the fitting jobs and write
    to results to disk
    """
    # create a pool, fit all files
    with tables.open_file(arguments.outname, mode="a") as gaussdec_file:
        gaussdec_table = gaussdec_file.root.gaussdec_ebhis

        pool = Pool(
            30,
            initializer=initializer,
            initarguments=(arguments.infile,))

        ebhis_store = tables.open_file(arguments.infile, mode="r", title="EBHIS")
        ebhis_table = ebhis_store.root.ebhis

        for row_index, fitresults in pool.imap(do_fit, get_row_index(arguments.nsamples, ebhis_table)):
            hpxindex = ebhis_table[row_index]['HPXINDEX']
            glon = ebhis_table[row_index]['GLON']
            glat = ebhis_table[row_index]['GLAT']

            for i in range(len(fitresults)/3):
                entry = gaussdec_table.row
                entry['hpxindex'] = hpxindex
                entry['glon'] = glon
                entry['glat'] = glat

                entry['amplitude'] = fitresults[i * 3]
                entry['center_c'] = fitresults[i * 3 + 1]
                entry['center_kms'] = chan2velo(entry['center_c'])

                entry['width_c'] = fitresults[i * 3 + 2]
                entry['width_kms'] = entry['width_c'] * CDELT3

                entry.append()

        gaussdec_table.flush()

    return 0


def main():
    """
    Create the argparser, create the tables and perform the fit
    """

    # evaluate parsed arguments
    argp = argparse.ArgumentParser(description=__doc__)

    argp.add_argument(
        '-s',
        '--survey',
        default='EBHIS',
        metavar='survey',
        choices=['GASS', 'EBHIS'],
        help='Survey that is used for the decomposition',
        type=str)

    argp.add_argument(
        '-i',
        '--infile',
        default='/vol/ebhis1/data1/dlenz/projects/ebhis2pytable/data/ebhis.h5',
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
        'outname',
        metavar='output_filename',
        type=str)

    argp.add_argument(
        '-c',
        '--clobber',
        default=False,
        metavar='clobber',
        help='clobber',
        type=bool)

    args = argp.parse_args()

    # check and create output h5file
    create_tables(args)

    # fit files
    fit_spectra(args)


# main
if __name__ == '__main__':
    main()
























