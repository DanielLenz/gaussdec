from typing import NamedTuple, List, Set, Dict, Callable, Iterator
from pathlib import Path
import logging

import matplotlib as mpl

mpl.use("pdf")
import matplotlib.pyplot as plt
import click
import numpy as np
import tables
import healpy as hp
from scipy import stats

from myhelpers import misc, hpx, plots
from myhelpers.datasets import hi4pi

import site

site.addsitedir(misc.bpjoin("gaussdec"))

from src import this_project as P
from src import core


def get_nhi_in(infile: Path, chan_min: int, chan_max: int) -> np.ndarray:
    with tables.open_file(infile) as s:
        logging.info(f"Opened {infile}, survey has shape {s.root.survey.shape}")
        nhi = np.nansum(s.root.survey[:, chan_min:chan_max], axis=1)
        nhi *= hi4pi.cK2nhi
        logging.info(f"Computed NHI, shape is {nhi.shape}")

    return nhi


def make_maps(config):
    #     quantities = [
    #             nhi_lvc_neg,
    #             nhi_ivc_neg,
    #             nhi_hvc_neg,
    #             nhi_lvc_pos,
    #             nhi_ivc_pos,
    #             nhi_hvc_pos,
    #             nhi_hvc,
    #             nhi_lvc,
    #             nhi_ivc,
    #             ]
    # Open store
    store = tables.open_file(config["paths"]["outfile"])
    table = store.root.gaussdec

    # Make maps
    nhi = core.reconstruct_coldens(table, full=False)
    nhi_in = get_nhi_in(
        infile=config["paths"]["infile"],
        chan_min=config["fit_parameters"]["v_range_channels"][0],
        chan_max=config["fit_parameters"]["v_range_channels"][1],
    )
    nhi_residual = nhi_in - nhi

    # Save
    hp.write_map(config["paths"]["procdir"].joinpath("nhi.hpx.fits"), nhi, **hpx.BASEKW)
    hp.write_map(
        config["paths"]["procdir"].joinpath("nhi_in.hpx.fits"), nhi_in, **hpx.BASEKW
    )
    hp.write_map(
        config["paths"]["procdir"].joinpath("nhi_residual.hpx.fits"),
        nhi_residual,
        **hpx.BASEKW,
    )


def plot_maps(config):
    # Plot
    plotters = build_plotters()
    for plotter in plotters:
        plot_nhi(nhi, plotter)
        plt.savefig(
            config["paths"]["plotdir"].joinpath(f"nhi_{plotter.suffix}.pdf"), dpi=300
        )

        plot_nhi(nhi_in, plotter)
        plt.savefig(
            config["paths"]["plotdir"].joinpath(f"nhi_in_{plotter.suffix}.pdf"), dpi=300
        )

        plot_nhi_residual(nhi_residual, plotter)
        plt.savefig(
            config["paths"]["plotdir"].joinpath(f"nhi_residual_{plotter.suffix}.pdf"),
            dpi=300,
        )

    store.close()
    return


@dataclass
class Plotter:
    func: Callable
    kwargs: Dict
    suffix: str


def build_plotters() -> Iterator(Plotter):
    """
    Constructs visualization functions such as hp.mollview and hp.cartview.

    Returns
    -------
    plotters : Iterator
        Plot functions, instances of Plotter
    """

    mw_view = Plotter(func=hp.mollview, kwargs=dict(), suffix="mw")
    ot_view = Plotter(func=hp.orthview, kwargs=dict(rot=[0, 90]), suffix="orth")

    lonras = [[10, 20], [40, 60]]
    latras = [[30, 40], [40, 60]]

    cart_views = [
        Plotter(
            func=hp.cartview, kwargs=dict(lonra=lonra, latra=latra), suffix=f"cart{i}"
        )
        for i, (lonra, latra) in enumerate(zip(lonras, latras))
    ]
    plotters = iter((mw_view, ot_view, *cart_views))
    return plotters


def plot_nhi(nhi, plotter):
    plotter.func(nhi, **plotter.kwargs)
    hp.graticule()


def plot_nhi_residual(nhi_residual, plotter):

    vmin, med, vmax = stats.scoreatpercentile(nhi_residual, [1, 50, 99])
    kwargs = dict(min=vmin, max=-1 * vmin, cmap="RdYlBu", **plotter.kwargs)
    plotter.func(nhi_residual, **kwargs)
    hp.graticule()


def make_histograms():
    pass


def make_spectra():
    pass


def analyze(config):
    make_maps(config)


@click.command()
@click.option("--modeldir", help="Model directory.")
def main(modeldir):
    # Set up logger
    logging.basicConfig(level=logging.INFO, **misc.LOGGING_KW)
    logging.info("Initialized logger")

    configfile = Path(modeldir).joinpath("decompose.yaml")
    config = misc.parse_config(configfile)
    analyze(config)


if __name__ == "__main__":
    main()
