from typing import List, Set
from pathlib import Path
import logging
import glob

import numpy as np
import healpy as hp
import tables
import click

from myhelpers import misc, hpx

import site

site.addsitedir(misc.bpjoin("gaussdec"))
from src import this_project as P


def merge_tables(modeldir: Path) -> None:
    # Set paths
    inpaths = map(Path, glob.glob(modeldir.joinpath("raw/basepix*.h5").as_posix()))
    outfile = modeldir.joinpath("processed/gaussdec.h5")

    tabs = [tables.open_file(inpath, "r") for inpath in inpaths]

    merged_store = tables.open_file(outfile, "w")
    merged_store.create_table(
        where="/",
        name="gaussdec",
        description=tabs[0].root.gaussdec.description,
        title="Gauss Decomposition",
    )

    for table in tabs:
        merged_store.root.gaussdec.append(table.root.gaussdec[:])

    # Close all stores
    merged_store.close()
    for tab in tabs:
        tab.close()

    return


@click.command()
@click.option("--modeldir", help="Model directory.")
def main(modeldir):
    # Set up logger
    logging.basicConfig(level=logging.INFO)

    # Set path
    modeldir = Path(modeldir)

    # Merge tables
    merge_tables(modeldir)

if __name__ == "__main__":
    main()
