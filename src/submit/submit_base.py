import os
import logging
import click
from glob import glob
from subprocess import run

from myhelpers import misc

@click.command()
@click.option("--modeldir", help="Model directory")
def main(modeldir):
    logging.basicConfig(level=logging.INFO, **misc.LOGGING_KW)
    logging.info(f"Fitting model in {modeldir}")

    workdir = os.environ["WORK"]

    files = glob(f"{workdir}/projects/gaussdec/data/indices/indices*.npy")
    n_files = len(files)

    logging.info(f"Total of {n_files} pixel files to process")

    for idx_number in range(n_files):
        run([
            "sbatch",
            f"{workdir}/projects/gaussdec/src/submit/run_single_skx.sh",
            f"{modeldir}",
            f"{idx_number}",
            ])

if __name__ == '__main__':
    main()
