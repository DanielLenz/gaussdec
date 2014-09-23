import argparse

from astropy.io import fits

from reconstruct_components import reconstruct
from fit_model import fit

if __name__ == '__main__':

    # evaluate parsed arguments
    argp = argparse.ArgumentParser(description='Create H2 maps based on the HI-dust correlation.')
    argp.add_argument("-s","--source",
                default='/vol/ebhis2/data1/dlenz/data/gaussdec/',
                metavar='filepath',
                help="Path to fitresults.",
                type=str)
    argp.add_argument("-r","--reconstruct",
                default='fixed',
                metavar='reconstruction_method',
                help="Method used for reconstruction.",
                type=str)
    argp.add_argument("-f","--fit",
                default='pooled',
                metavar='fitting_method',
                help="Method used for fitting the model.",
                type=str)
    argp.add_argument('outname',
                metavar='output_filename',
                type=str)
    args = argp.parse_args()

    # reconstruct components
    components = reconstruct(args.source, method=args.reconstruct)
    for k, v in components.iteritems():
        fits.writeto(args.source + k + '.hpx.fits', v, clobber=True)

    mcmc_model = fit(components, args.source, method=args.fit)
    # fit_model
    # fit(components)














