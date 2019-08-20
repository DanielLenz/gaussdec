"""
We compare different backends to fit a number of spectra.

Backends:
- theano
- numba
- Pure python/numpy

For each, we fit several sightlines (randomized or list of hpx indices)
and output
- Spectra (including the different components)
- All the metrics/additional information as a textbox

"""

from typing import Dict, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass

import healpy as hp
import tables
import numpy as np

# import logging

from myhelpers import misc

import site

site.addsitedir(misc.bpjoin("gaussdec"))

from src.configure.build_config import Config
from src.analysis import analyze
from src.decompose import call_specfit as cs

def run_numba():
    pass


@dataclass
class Arguments:
    """Mimics the argparse arguments passed to call_specfit.py"""

    infile: str=misc.bpjoin("HI4PI/data/raw/survey.npy")
    config_file: str
    nsamples: int=-1
    hpxindices: Optional[str]=None
    clobber: Optional[bool]=True
    outname: str

def run_theano():
    # Build the arguments object to mimic the argparse object 
    arguments = Arguments(
        config_file=...,
        hpxindices=hpxindex_file,
        outname=outname,
    )
    cs.fit_all_spectra(arguments=arguments)

    return

def run_all(*args, **kwargs):
    run_numba(*args, **kwargs)
    run_theano(*args, **kwargs)

    return


def main():
    # Set output base directory
    # E.g.
    # gaussdec/models/compare_backends/numba/
    # Each has a data and figures subfolder
    outpath = misc.PPATH.joinpath('models/compare_backends/')
    outpath.mkdir(exist_ok=True, parents=True)

    # Build config
    config = Config(modeldir=outpath)
    config.to_yaml()

    # Load the hpxindices to fit
    n_samples = 10
    hpxindices = cs.get_row_index(
        n_samples=n_samples,
        hpxindices=None,
        n_spectra_in_survey=g_npix,
    )

    indices_filepath = outpath.joinpath("raw/hpxindices.npy")
    np.save(indices_filepath, hpxindices)

    # Call all backends, or only selected ones
    run_theano()

    return

if __name__ == "__main__":
    g_nside = 1024
    g_npix = hp.nside2npix(g_nside)
    main()