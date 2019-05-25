import argparse
from pathlib import Path

import matplotlib as mpl

mpl.use("pdf")
import matplotlib.pyplot as plt
import tables
import numpy as np
import healpy as hp

from myhelpers.datasets import hi4pi
from myhelpers import misc

import site

site.addsitedir(misc.bpjoin("gaussdec"))
from src.decompose import specfitting
from src.analysis import core

"""
Inspect the Gaussian decomposition of EBHIS and GASS
"""

"""
Functions
---------

inspect_spectra(data_table, model_table, nsamples) : Inspect a given, random
    number of spectra
"""


def inspect_spectra(data_table, model_table, nsamples, x_model):
    """
    Inspect a given, random number of spectra
    """
    model_functions = specfitting.make_multi_gaussian_model()
    f_model = model_functions[0]

    # draw random, unique hpxindices
    indices = np.unique(model_table.cols.hpxindex[:])
    sample_indices = np.random.choice(indices, size=nsamples, replace=False)

    spectra = []
    model_spectra = []
    for sample_index in sample_indices:
        # data
        spectra.append(np.squeeze(data_table[sample_index]))

        # model
        gauss_params = np.array(
            [
                [row["amplitude"], row["center_kms"], row["sigma_kms"]]
                for row in model_table.where("""hpxindex=={}""".format(sample_index))
            ]
        )
        model_spectra.append(
            CDELT3 / 1.0e3 * f_model(gauss_params.flatten(), x_model)[1]
        )

    return spectra, model_spectra


def make_maps(gdec, outpath: Path):
    # Reconstruct column density
    hi_model = core.reconstruct_coldens(table=gdec)

    # Inspect reconstruction
    hp.mollview(hi_model)
    plt.savefig(outpath.joinpath("model_mollview.pdf"), dpi=300)

    # number of components
    n_comps = core.make_ncomp_map(table=gdec)
    hp.mollview(n_comps)
    plt.savefig(outpath.joinpath("ncomps_mollview.pdf"), dpi=300)


def main():
    """
    Inspect the Gaussian decomposition
    """

    # evaluate parsed arguments
    argp = argparse.ArgumentParser(description=__doc__)

    argp.add_argument(
        "-d",
        "--data",
        default=misc.bpjoin("HI4PI/data/raw/HI4PI_DR1.h5"),
        metavar="infile",
        help="Data pytable",
        type=str,
    )

    argp.add_argument(
        "-n",
        "--nsamples",
        default=5,
        help="Number of random sightlines that are inspected",
        type=int,
    )

    argp.add_argument(
        "gaussdec", help="location of the Gaussian decomposition", type=str
    )

    args = argp.parse_args()

    # Load tables
    gdec_store = tables.open_file(args.gaussdec)
    gdec = gdec_store.root.gaussdec

    data_store = tables.open_file(args.data)
    data = data_store.root.survey

    outpath = Path(misc.bpjoin("gaussdec/figures/"))
    outpath.mkdir(parents=True, exist_ok=True)

    # Make maps
    make_maps(gdec=gdec, outpath=outpath)

    # velocity axis in km/s


#     velos_model = np.linspace(-500.0, 500.0, 1e4)
#
#     spectra, model_spectra = inspect_spectra(
#         data_table=data, model_table=gdec, nsamples=10, x_model=x_model
#     )
#
#     shift = 0
#     for i, (spectrum, model_spectrum) in enumerate(zip(spectra, model_spectra)):
#         pl.plot(hi4pi.VELOGRID, spectrum + shift)
#         pl.plot(velos_model, model_spectrum + shift)
#         shift += np.nanmax(spectrum)
#
#     pl.show()


if __name__ == "__main__":
    main()
