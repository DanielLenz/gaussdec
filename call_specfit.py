from functools import partial
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

# ebhis standard spectral restframe
CRPIX3 = 471.921630003202
CDELT3 = 1288.23448620083
chan2velo = lambda c: (c - CRPIX3) * CDELT3


def create_tables(args):

    # append to an existing h5file
    if (os.path.exists(args.outname)):
        warnings.warn('File {} already exists!'.format(args.outname))

        return

    # create a new h5file and append entries
    else:
        print 'Creating file {} ...'.format(args.outname)

        class GaussDec(tables.IsDescription):
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
    
    with tables.open_file(args.outname, mode="w", title="Gaussdec") as fobj:
            ebhis = fobj.create_table(fobj.root, 'gaussdec_ebhis', GaussDec, "Gauss decomposition EBHIS")
            ebhis.cols.hpxindex.create_csindex()
            ebhis.autoindex = True

            # gass = fobj.create_table(fobj.root, 'gaussdec_gass', GaussDec, "Gauss decomposition GASS")
            # gass.cols.hpxindex.create_csindex()
            
    return 0


def initializer(infile):
    
    global f_model, f_residual, f_objective, f_jacobian, f_stats

    # create theano functions
    f_model, f_residual, f_objective, f_jacobian, f_stats = make_multi_gaussian_model()

    global ebhis_file
    ebhis_file = tables.open_file(infile, mode="r", title="EBHIS")


def do_fit(row_index):

    row = ebhis_file.root.ebhis[row_index]
    fitresults = fit_spectrum(row['DATA'], f_objective, f_jacobian, f_stats, default_p)['parameters']

    return row_index, fitresults


def get_row_index(table):
    for row_index in range(table.nrows):
        if not (row_index % 10000):
            print 'Working on row {i} of {n}...'.format(i=row_index, n=table.nrows)
        yield row_index


def fit_spectra(args):

    # create a pool, fit all files
    with tables.open_file(args.outname, mode="a", title="Gaussdec") as gaussdec_file:
        gaussdec_table = gaussdec_file.root.gaussdec_ebhis
        
        p = Pool(32, initializer=initializer, initargs=(args.infile,))
        
        ebhis_smile = tables.open_file(args.infile, mode="r", title="EBHIS")
        ebhis_table = ebhis_smile.root.ebhis

        for row_index, fitresults in p.imap(do_fit, get_row_index(ebhis_table)): 
            
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
                entry['width_kms'] =  entry['width_c'] * CDELT3

                entry.append()
                        
        gaussdec_table.flush()

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
    argp.add_argument('-i', '--infile',
                default='/vol/ebhis2/data1/dlenz/projects/ebhis2pytable/data/ebhis.h5',
                metavar='infile',
                help='Source pytable.',
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
    fit_spectra(args)
















