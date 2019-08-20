"""Allow to decompose and to plot individual lines of sight to debug and fine tune
the fits.
"""

from typing import Dict, List, Optional, Tuple
from pathlib import Path
from subprocess import run
from tempfile import TemporaryDirectory

import theano
import healpy as hp
import tables
import numpy as np

# import logging

from myhelpers import misc
from myhelpers.datasets import hi4pi

import site

site.addsitedir(misc.bpjoin("gaussdec"))
from src.decompose import call_specfit as cs
from src.configure.build_config import Config
from src.analysis import analyze


def debug(sightlines: List[int], nside: int = 1024):
    """
    1) Plot spectra (full range!)
    2) Print comprehensive overview of fit parameters
    """
    with TemporaryDirectory() as tmpdir:

        # Build config
        tmpdir_path = Path(tmpdir)
        config = Config(modeldir=tmpdir_path)
        config.to_yaml()

        # gdec_filename = tmpdir_path.joinpath("processed/gaussdec.h5")
        gdec_filename = config.config['paths']['outfile']

        # Build file for indices
        indices_filepath = tmpdir_path.joinpath("hpxindices.npy")
        np.save(indices_filepath, sightlines)

        # Decompose
        run(
            [
                "python3",
                "src/decompose/call_specfit.py",
                "--config",
                tmpdir_path.joinpath("decompose.yaml"),
                "--hpxindices",
                indices_filepath,
                "--clobber",
                "True",
                gdec_filename,
            ]
        )

        # Plot spectra
        outdir = misc.PPATH.joinpath("gaussdec/figures/debugging/")
        grid_shape = (len(sightlines), 1)
        spectra = analyze.make_spectra(config.config, grid_shape=grid_shape)
        analyze.plot_spectra(outdir=outdir, spectra=spectra, grid_shape=grid_shape, vlim=None)

    return


def run_fit():
    pass


def main():
    """Provide hpxindices of individual lines of sight.
    Use a temporary directory to store the result of the decomposition?
    """

    # Set sightlines
    sightlines = [10682996, 10679095, 10675191]
    nside = 1024

    # Debug!
    debug(sightlines, nside=nside)

    return


if __name__ == "__main__":
    main()
