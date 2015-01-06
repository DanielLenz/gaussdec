import tables
import numpy as np
import pylab as pl

from astropy.io import fits
import healpy as hp

from specfitting import make_multi_gaussian_model

# convert channels to m/s
CRPIX3 = 471.921630003202
CDELT3 = 1288.23448620083
chan2velo = lambda c: (c - CRPIX3) * CDELT3

def reconstruct_coldens(table):

    npix = hp.nside2npix(2**10)
    hi_model = np.zeros(npix, dtype=np.float32)
    
    for row in table:
        hi_model[row['hpxindex']] += row['amplitude']

    # convert to cm**-2, 1.288 is EBHIS chanwidth
    to_coldens = 1.82e18 * 1.288
    
    return hi_model * to_coldens


def make_ncomp_map(table):

    npix = hp.nside2npix(2**10)
    ncomps = np.zeros(npix, dtype=np.float32)
    
    for row in table:
        ncomps[row['hpxindex']] += 1

    return ncomps


def inspect_spectra(data_table, model_table, nsamples, x_model):

    f_model, f_residual, f_objective, f_jacobian, f_stats = make_multi_gaussian_model()
    
    # draw random, unique hpxindices
    indices = model_table.cols.hpxindex[:]
    sample_indices = np.random.choice(np.unique(indices), size=nsamples, replace=False)

    spectra = []
    model_spectra = []
    
    for sample_index in sample_indices:
        # data
        spectra.append(np.squeeze(data_table.read_where("""HPXINDEX=={}""".format(sample_index))['DATA']))
        
        # model
        gauss_params = np.array([[row['amplitude'], row['center_kms'], row['width_kms']] for row in model_table.where("""hpxindex=={}""".format(sample_index))])
        model_spectra.append(f_model(gauss_params.flatten(), x_model)[1])

    return spectra, model_spectra


    # read spectra from data and evalutate model
    x = np.linspace(0., 945., 945)
    with fits.open('data/ebhis_hpx_1024_hpxpix015of192_G7.fits') as hdulist:
        hpxfile = hdulist[1].data

        spectra = []
        model_spectra = []
        for sample_index in sample_indices:
            # data
            spectra.append(np.squeeze(hpxfile['DATA'][hpxfile['HPXINDEX'] == sample_index]))
            # model
            gauss_params = np.array([[row['amplitude'], row['center_c'], row['width_c']] for row in table.where("""index=={}""".format(sample_index))])
            model_spectra.append(f_model(gauss_params.flatten(), x)[1])

    return spectra, model_spectra



def inspect_spectra(data_table, model_table, nsamples):

    return 0


def main():
    gaussdec_file = tables.open_file('data/gaussdec.h5', mode="r", title="Gaussdec")
    gaussdec_table = gaussdec_file.root.gaussdec_ebhis

    ebhis_file = tables.open_file(
        '/vol/ebhis2/data1/dlenz/projects/ebhis2pytable/data/ebhis.h5',
        mode="r",
        title="EBHIS")
    ebhis_table = ebhis_file.root.ebhis

    # inspect reconstruction
    hi_model = reconstruct_coldens(table=gaussdec_table)
    hp.mollview(hi_model, xsize=4000.)

    # inspect spectra
    x_data = chan2velo(np.arange(945))
    x_model = np.linspace(-500.e3, 500.e3, 1e4)

    spectra, model_spectra = inspect_spectra(data_table=ebhis_table, model_table=gaussdec_table, nsamples=10, x_model=x_model)
    
    shift = 0
    for i, (spectrum, model_spectrum) in enumerate(zip(spectra, model_spectra)):
        pl.plot(x_data, spectrum + shift)
        pl.plot(x_model, model_spectrum + shift)
        shift += np.nanmax(spectrum) 

    pl.show()


if __name__ == '__main__':
    main()








