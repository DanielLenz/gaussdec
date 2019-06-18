from pathlib import Path
import logging

import yaml
import numpy as np

from myhelpers import misc
from myhelpers.datasets import hi4pi


class Config:
    """
    Fitparameters
    Paths
    submit
    """

    def __init__(self, modeldir):
        self.config = {}
        self.modeldir = modeldir

        self.set_fitparameters()
        self.set_paths()
        self.set_submit()
        self.mk_dirs()

    def set_fitparameters(self) -> None:
        velo_min, velo_max = -300, 300  # in km/s
        self.config["fit_parameters"] = dict()
        self.config["fit_parameters"]["v_range_kms"] = (velo_min, velo_max)
        self.config["fit_parameters"]["v_range_channels"] = (
            hi4pi.velo2channel(velo_min),
            hi4pi.velo2channel(velo_max),
        )
        self.config["fit_parameters"]["min_components"] = 1
        self.config["fit_parameters"]["max_components"] = 12
        self.config["fit_parameters"]["iterations"] = 8
        self.config["fit_parameters"]["int_low"] = 5e18 / 1.82e18 / 1.28
        self.config["fit_parameters"]["int_high"] = 5e21 / 1.82e18 / 1.28
        self.config["fit_parameters"]["sigma_low"] = np.sqrt(50 / 21.85) / 1.28 / 2.35
        self.config["fit_parameters"]["sigma_high"] = (
            np.sqrt(40_000 / 21.85) / 1.28 / 2.35
        )
        self.config["fit_parameters"]["pdf_threshold"] = 0.05
        self.config["fit_parameters"]["pdf_kernel"] = 3.32
        self.config["fit_parameters"]["fit_method"] = "l-bfgs-b"
        self.config["fit_parameters"]["trim"] = 200
        self.config["fit_parameters"]["iteration_size"] = 5

    def set_paths(self) -> None:
        self.config["paths"] = dict()
        self.config["paths"]["infile"] = Path(
            misc.bpjoin("HI4PI/data/raw/HI4PI_DR1.h5")
        )
        self.config["paths"]["modeldir"] = self.modeldir

        self.config["paths"]["rawdir"] = self.modeldir.joinpath("raw/")
        self.config["paths"]["procdir"] = self.modeldir.joinpath("processed/")
        self.config["paths"]["plotdir"] = self.modeldir.joinpath("figures/")

        self.config["paths"]["outfile"] = self.modeldir.joinpath(
            "processed/gaussdec.h5"
        )

    def mk_dirs(self) -> None:
        for kind in ["rawdir", "procdir", "plotdir"]:
            self.config["paths"][kind].mkdir(parents=True, exist_ok=True)

    def set_submit(self) -> None:
        self.config["submit"] = dict()

    def to_yaml(self):
        out_filename = self.modeldir.joinpath("decompose.yaml")

        with open(out_filename, "w") as f:
            logging.info(f"Writing config file to {out_filename}.")
            yaml.dump(self.config, f, default_flow_style=False)


def main():
    logging.basicConfig(level=logging.INFO)
    modeldir = Path(misc.bpjoin("gaussdec/models/complete2/"))
    config = Config(modeldir=modeldir)

    # Write to yaml
    config.to_yaml()


if __name__ == "__main__":
    main()
