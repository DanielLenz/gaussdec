from pathlib import Path

import numpy as np
import healpy as hp

from myhelpers import misc


def build_hpx_basepix(nside_high: int, nside_low: int, outpath: Path):
    npix_low = hp.nside2npix(nside_low)
    values = np.arange(npix_low)
    values = hp.ud_grade(values, nside_high)

    for idx in range(npix_low):
        indices = np.where(values == idx)[0]
        np.save(outpath.joinpath(f"indices{idx}"), indices)


def build_randomized(nside: int, outpath: Path, n_chunks: int = 22):
    npix = hp.nside2npix(nside)
    indices = np.arange(npix)
    np.random.shuffle(indices)

    list_of_chunks = np.array_split(indices, n_chunks)

    for i, chunk in enumerate(list_of_chunks):
        np.save(outpath.joinpath(f"indices{i}"), chunk)


if __name__ == "__main__":
    nside = 1024

    outpath = Path(misc.bpjoin("gaussdec/data/indices/"))
    outpath.mkdir(parents=True, exist_ok=True)

    build_randomized(nside=nside, outpath=outpath, n_chunks=25)
