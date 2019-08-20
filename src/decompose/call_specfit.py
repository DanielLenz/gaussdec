from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import multiprocessing as mp
import argparse
from functools import partial, reduce
import operator
import os

import theano
import healpy as hp
import tables
import numpy as np

# import logging

from myhelpers import misc
from myhelpers.datasets import hi4pi

import site

site.addsitedir(misc.bpjoin("gaussdec"))
from src.decompose.specfitting import fit_spectrum, make_multi_gaussian_model

"""
Generate a Gaussian decomposition of spectra, based on and written to hd5-files
"""


class GaussDec(tables.IsDescription):
    """
    Description for the pytable, specifying the columns
    and their data types """

    # coordinates
    hpxindex = tables.Int32Col()
    glon = tables.Float32Col()
    glat = tables.Float32Col()

    # Gauss fit parameters
    line_integral_cK = tables.Float32Col()
    line_integral_kmsK = tables.Float32Col()
    peak_amplitude = tables.Float32Col()

    center_c = tables.Float32Col()
    center_kms = tables.Float32Col()

    sigma_c = tables.Float32Col()
    sigma_kms = tables.Float32Col()


# A work chunk consists of a list of hpx indices, and the corresponding spectra
WORK_CHUNK = Tuple[List[int], List[np.ndarray]]

# Hardcoding the nside, very unlikely to change
NSIDE = 1024
NPIX = hp.nside2npix(NSIDE)


def create_tables(outname: Path, clobber: bool):
    """
    If the table exists, abort. Else, create a new hdf5-table where the
    decomposition is saved.
    """
    # logging.info(f"Creating output table, outname is {arguments.outname}.")

    # Read or create file
    if os.path.isfile(outname) and not clobber:
        raise IOError("File already exists")
    else:
        print("Creating file {}".format(outname))

    store = tables.open_file(outname, mode="w")

    # Check for existing tables
    gaussdec = store.create_table(
        store.root, "gaussdec", GaussDec, "Gauss decomposition"
    )
    gaussdec.cols.hpxindex.create_csindex()
    gaussdec.autoindex = True

    return 0


def get_row_index(
    n_samples: int, hpxindices: Optional[Union[str, Path]], n_spectra_in_survey: int
):
    """
    Yield all the rows of the input file or a randomly chosen sample
    """
    if hpxindices is None:
        if n_samples < 0:
            for row_index in range(n_spectra_in_survey):
                yield row_index
        else:
            sample_indices = np.random.randint(
                low=0, high=n_spectra_in_survey, size=n_samples
            )
            for row_index in sample_indices:
                yield row_index
    else:
        indices = np.load(hpxindices)
        for row_index in indices:
            yield row_index


def workerresults2dict(worker_results: Dict, row_index: int) -> List[Dict]:
    """
    For a single line of sight, convert the raw results coming from the worker into a list of dictionaries

    Example
    -------
    worker_results = [amp1, center1, sigma1, amp2, center2, sigma2]
    row_index = 11231

    result_dict = [{'hpxindex': 11231, 'amp1': amp1, ...}, {'hpxindex': 11231, 'amp2': amp2, ... }]

    """
    result_dicts = []

    glon, glat = hp.pix2ang(NSIDE, row_index, lonlat=True)

    resulting_parameters = worker_results["parameters"]

    n_components = len(resulting_parameters) // 3

    for i in range(n_components):
        entry = dict()
        entry["hpxindex"] = row_index
        entry["glon"] = glon
        entry["glat"] = glat

        entry["line_integral_cK"] = resulting_parameters[i * 3]
        entry["line_integral_kmsK"] = entry["line_integral_cK"] * hi4pi.CDELT3
        entry["center_c"] = resulting_parameters[i * 3 + 1]
        entry["center_kms"] = hi4pi.channel2velo(entry["center_c"])

        entry["sigma_c"] = resulting_parameters[i * 3 + 2]
        entry["sigma_kms"] = entry["sigma_c"] * hi4pi.CDELT3

        # Peak of the component in Kelvin
        # Peak = Integral / 2pi / sigma
        entry["peak_amplitude"] = (
            entry["line_integral_cK"] / 2.0 / np.pi / entry["sigma_c"]
        )

        result_dicts.append(entry)

    return result_dicts


def fit_chunk(chunk: WORK_CHUNK, parameters=None):
    """
    Work is performed on one cpu. The worker gets a workload (typically workload / n_nodes / n_cpus)
    and does all the necessary setup.
    """
    # We only import theano at the worker level to avoid the theano compile lock

    # Each work chunk consists of the hpx indices and the corresponding spectra
    row_indices, spectra = chunk

    # Build theano functions
    f_model, f_residual, f_objective, f_jacobian, f_stats = make_multi_gaussian_model()

    # Fit all spectra in simple map-fashion
    work_function = partial(
        fit_spectrum,
        objective=f_objective,
        jacobian=f_jacobian,
        stats=f_stats,
        p=parameters,
    )
    raw_results = map(work_function, spectra)

    # result_dicts = map(workerresults2dict, *zip(raw_results, row_indices))

    result_dicts = [
        workerresults2dict(result, index)
        for result, index in zip(raw_results, row_indices)
    ]

    # The result_dicts are of type List[List[Dict]], we convert this to a flat list of dicts (aka List[Dict])
    result_dicts = reduce(operator.iconcat, result_dicts, [])

    return result_dicts


def fit_all_spectra(arguments):
    """
    for row_index, fitresults in pool.imap(
        # for row_index, fitresults in map(
        do_fit_eff,
        get_row_index(arguments.nsamples, arguments.hpxindices, n_spectra_in_survey),
        #            chunksize=10_000,
    ):
    """
    # Parse config. This creates a dict object, based on the
    # yaml file
    config: Dict = misc.parse_config(arguments.config)

    # Get the full workload, i.e. the row indices
    # hpx_indices = np.load(arguments.hpxindices)
    hpx_indices = list(
        get_row_index(
            arguments.nsamples, arguments.hpxindices, n_spectra_in_survey=NPIX
        )
    )

    # Create a Pool
    pool = mp.Pool()

    # Build the iterator that delivers the chunks to the workers
    # Each chunk consists of the indices and the spectra
    chunks = get_chunks(input_filename=arguments.infile, indices=hpx_indices)

    # map chunks to workers. Each worker deals with 1/n_cpus of the load
    fit_chunk_eff = partial(fit_chunk, parameters=config["fit_parameters"])
    result_dicts = pool.imap(fit_chunk_eff, chunks)

    # The result_dicts are of type List[List[Dict]], we convert this to a flat list of dicts (aka List[Dict])
    result_dicts = list(reduce(operator.iconcat, result_dicts, []))

    return result_dicts


def get_chunks(
    input_filename: Path, indices: np.ndarray, n_chunks: Optional[int] = None
):
    # Read full file
    # memmap might cause issues, hence we load the full file
    survey = np.load(input_filename)

    # Set n_chunks to n_cpus if it is not provided
    if n_chunks is None:
        n_chunks = mp.cpu_count()

    # Slice the data along the first axis, yield one chunk at a time. Also yield the row number
    for index_chunk in np.array_split(indices, n_chunks):
        data_chunk = survey[index_chunk]
        yield index_chunk, data_chunk


def save_fitresults(out_filename, fitresult_dicts: List[Dict]) -> None:
    """
    Organize the output and put it into a table. The fitresults are passed as a list of dicts,
    and is then put into an HDF5 table with one row for each of these dicts.
    """

    # Put it into a pytable
    with tables.open_file(out_filename, mode="a") as gdec_store:
        gdec_table = gdec_store.root.gaussdec

        for fitresult_dict in fitresult_dicts:
            row = gdec_table.row

            for key, value in fitresult_dict.items():
                row[key] = value
            row.append()

    return


def main():
    """
    Create the argparser, create the tables and perform the fit
    """

    # evaluate parsed arguments
    argp = argparse.ArgumentParser(description=__doc__)

    argp.add_argument(
        "-i",
        "--infile",
        default=misc.bpjoin("HI4PI/data/raw/survey.npy"),
        metavar="infile",
        help="Source pytable",
        type=str,
    )

    argp.add_argument(
        "-p", "--config", metavar="config_file", help="Configuration file", type=str
    )

    argp.add_argument(
        "-n",
        "--nsamples",
        default=-1,
        metavar="nsamples",
        help="Number of random sightlines that are fitted",
        type=int,
    )

    argp.add_argument(
        "-x",
        "--hpxindices",
        default=None,
        metavar="hpxindices",
        help="Location of a npy file that contains the hpx indices to fit",
        type=str,
    )

    argp.add_argument(
        "-c", "--clobber", default=False, metavar="clobber", help="clobber", type=bool
    )

    argp.add_argument("outname", metavar="output_filename", type=str)

    args = argp.parse_args()

    # check and create output h5file
    create_tables(outname=args.outname, clobber=args.clobber)

    # fit files
    fitresult_dicts = fit_all_spectra(args)

    # Save output
    save_fitresults(args.outname, fitresult_dicts)


# main
if __name__ == "__main__":
    # Initialize logger
    # logging.basicConfig(level=logging.INFO, format=misc.LOGGING_KW)
    # theano.gof.compilelock.set_lock_status(False)

    main()
