from typing import NamedTuple, List, Set, Dict, Callable, Iterator, Tuple, Optional
from pathlib import Path
from functools import reduce
import operator
from collections import deque
import itertools as it
import logging
from dataclasses import dataclass

import matplotlib as mpl

mpl.use("pdf")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import click
import numpy as np
import tables
import healpy as hp
from scipy import stats

from myhelpers import misc, hpx, plots
from myhelpers.datasets import hi4pi

import site

site.addsitedir(misc.bpjoin("gaussdec"))

# from src.decompose.specfitting import make_multi_gaussian_model
from src import this_project as P
from src import core

# Globals
g_backend = "pdf"


def get_nhi_in(infile: Path, chan_min: int, chan_max: int) -> np.ndarray:
    survey = np.load(infile, mmap_mode="r")
    logging.info(f"Opened {infile}, survey has shape {survey.shape}")

    nhi = np.nansum(survey[:, chan_min:chan_max], axis=1)
    nhi *= hi4pi.cK2nhi
    logging.info(f"Computed NHI, shape is {nhi.shape}")

    return nhi


def gauss(channels: np.ndarray, amp: float, x0: float, sigma: float):
    """
    Evaluates a Gaussian with given parameters and x-coordinates.
    Can be either in in channels or in velocities, as long as units are
    consistent.
    """
    values = (
        amp
        / np.sqrt(2.0 * np.pi)
        / sigma
        * np.exp(-0.5 * (channels - x0) ** 2 / sigma ** 2)
    )
    return values


def make_maps(config) -> None:
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

    # Make maps
    # NHI out
    nhi = core.reconstruct_coldens(gaussdec_path=config["paths"]["outfile"], full=False)

    # NHI in
    nhi_in = get_nhi_in(
        infile=config["paths"]["infile"],
        chan_min=config["fit_parameters"]["v_range_channels"][0],
        chan_max=config["fit_parameters"]["v_range_channels"][1],
    )

    # NHI residual
    nhi_residual = nhi_in - nhi

    # Number of components
    with tables.open_file(config["paths"]["outfile"]) as store:
        table = store.root.gaussdec
        n_comp_map = core.make_ncomp_map(table)

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
    hp.write_map(
        config["paths"]["procdir"].joinpath("n_components.hpx.fits"),
        n_comp_map,
        **hpx.BASEKW,
    )

    return


def plot_maps(config):
    # Load maps
    map_names = ("nhi", "nhi_in", "nhi_residual", "n_components")
    nhi_out, nhi_in, nhi_residual, n_comp_map = (
        hp.read_map(
            config["paths"]["procdir"].joinpath(f"{map_name}.hpx.fits"), verbose=False
        )
        for map_name in map_names
    )

    # Plot
    plotters = build_plotters()
    for plotter in plotters:
        # NHI out
        plot_nhi(nhi_out, plotter)
        plt.savefig(
            config["paths"]["plotdir"].joinpath(f"nhi_{plotter.suffix}.{g_backend}"),
            dpi=300,
        )

        # NHI in
        plot_nhi(nhi_in, plotter)
        plt.savefig(
            config["paths"]["plotdir"].joinpath(f"nhi_in_{plotter.suffix}.{g_backend}"),
            dpi=300,
        )

        # NHI residual
        plot_nhi_residual(nhi_residual, plotter)
        plt.savefig(
            config["paths"]["plotdir"].joinpath(
                f"nhi_residual_{plotter.suffix}.{g_backend}"
            ),
            dpi=300,
        )

        # Number of components
        plot_n_comps(n_comp_map, plotter)
        plt.savefig(
            config["paths"]["plotdir"].joinpath(
                f"n_comps_{plotter.suffix}.{g_backend}"
            ),
            dpi=300,
        )

    return


@dataclass
class ImagePlotter:
    func: Callable
    kwargs: Dict
    suffix: str


def build_plotters() -> Iterator:
    """
    Constructs visualization functions such as hp.mollview and hp.cartview.

    Returns
    -------
    plotters : Iterator
        Plot functions, instances of ImagePlotter
    """

    universal_kw = dict(title="")
    mw_view = ImagePlotter(func=hp.mollview, kwargs=dict(**universal_kw), suffix="mw")
    ot_view = ImagePlotter(
        func=hp.orthview, kwargs=dict(rot=[0, 90], **universal_kw), suffix="orth"
    )

    lonras = [[10, 20], [40, 60]]
    latras = [[30, 40], [40, 60]]

    cart_views = [
        ImagePlotter(
            func=hp.cartview,
            kwargs=dict(lonra=lonra, latra=latra, **universal_kw),
            suffix=f"cart{i}",
        )
        for i, (lonra, latra) in enumerate(zip(lonras, latras))
    ]
    plotters = iter((mw_view, ot_view, *cart_views))
    return plotters


def plot_nhi(nhi, plotter):
    unit = r"$N_{\rm HI}\ [\rm cm^{-2}]$"

    plotter.func(nhi, unit=unit, **plotter.kwargs)
    hp.graticule()


def plot_nhi_residual(nhi_residual, plotter):
    unit = r"$\Delta N_{\rm HI}\ [\rm cm^{-2}]$"
    # vmin, med, vmax = stats.scoreatpercentile(nhi_residual, [1, 50, 99])
    vmin, vmax = -5.0e19, 5.0e19
    kwargs = dict(min=vmin, max=-1 * vmin, unit=unit, cmap="RdYlBu", **plotter.kwargs)
    plotter.func(nhi_residual, **kwargs)
    hp.graticule()


def plot_n_comps(n_comp_map, plotter):
    unit = r"$\# \rm components$"
    plotter.func(n_comp_map, unit=unit, **plotter.kwargs)
    hp.graticule()


def get_components_at_index(table, idx: int) -> List[Dict]:
    """
    For a given HEALPix index, create a list of all components parameters.
    Each component is represented as a dictionary.
    """
    column_names = table.colnames

    gauss_params = [
        {
            col_name: parameter_value
            for col_name, parameter_value in zip(column_names, row[:])
        }
        for row in table.where("""hpxindex=={}""".format(idx))
    ]

    return gauss_params


def build_components(fit_parameters: List[Dict], channels: np.ndarray):
    """
    Build the different components for a single line of sight,
    based on the fit parameters and the model config.

    Returns
    -------

    components : np.ndarray, shape=(n_components, n_channels)
        Spectra of all components for a line of sight
    total : np.ndarray, shape=(n_channels)
        Sum of all components
    """

    components = np.array(
        [
            gauss(
                channels,
                amp=param["line_integral_cK"],
                x0=param["center_c"],
                sigma=param["sigma_c"],
            )
            for param in fit_parameters
        ]
    )
    total = components.sum(axis=0)

    return components, total


@dataclass
class Spectrum:
    """
    This dataclass contains all the necessary information for a plot of the input
    and model spectrum.
    """

    velos: np.ndarray
    input_data: np.ndarray
    components: Tuple  # Tuple of the individual components Nx(line_integral_cK, center_c, sigma_c)
    total_model: np.ndarray
    hpxindex: int
    glon: float
    glat: float

    @property
    def n_components(self):
        return len(self.components)

    @staticmethod
    def generate_random(
        model_table: tables.Table, input_survey: np.ndarray, config: Dict
    ):
        # unique_indices = np.unique(model_table.cols.hpxindex)
        # idx = np.random.choice(unique_indices)
        idx = np.random.choice(model_table.cols.hpxindex)

        # Get coordinates
        glon, glat = hp.pix2ang(P.NSIDE, idx, lonlat=True)

        # Get velocities
        channel_slice = slice(*config["fit_parameters"]["v_range_channels"])
        velos = hi4pi.VELOGRID[channel_slice]
        channels = np.arange(*config["fit_parameters"]["v_range_channels"])

        # Get fit parameters
        parameter_dicts = get_components_at_index(model_table, idx)
        components, total_model = build_components(parameter_dicts, channels)

        # Get input data
        input_spectrum = input_survey[idx, channel_slice]

        # Build spectrum object
        spectrum = Spectrum(
            velos=velos,
            input_data=input_spectrum,
            components=components,
            total_model=total_model,
            hpxindex=idx,
            glon=glon,
            glat=glat,
        )

        return spectrum


# def get_set_at_indices(table, idx: int):
#     """
#     For a given HEALPix index, return a list of all the samples.
#     """
#     unique_indices = np.unique(table.cols.hpxindex[:])
#     sample_indices = np.random.choice(unique_indices, n_samples, replace=False)

#     # component_dict = {idx: [] for idx in indices}

#     return gauss_params


def make_spectra(config: Dict, grid_shape=(5, 3)):
    """
    Generate a grid of input spectra and their respective models.
    """
    n_samples = grid_shape[0] * grid_shape[1]

    with tables.open_file(
        config["paths"]["procdir"].joinpath("gaussdec.h5")
    ) as model_store:

        # Model PyTable
        model_table = model_store.root.gaussdec

        # Input data
        input_survey = np.load(config["paths"]["infile"], mmap_mode="r")

        # Build all the Spectrum objects
        spectra = [
            Spectrum.generate_random(model_table, input_survey, config)
            for _ in range(n_samples)
        ]

    return spectra
    # Open input store

    # Open model store

    # Load random samples from the input store
    # samples_for_indices = get_random_samples(model_table, n_samples=n_samples)

    # # Generate instances of Spectrum, based on the set of indices
    # spectra = map(generate_spectra, samples_for_indices)

    # # Map random samples to plot function
    # # Individual plot function takes (config, Spectrum)
    # for spectrum in spectra:
    #     plot_spectrum(Spectrum, config)
    # # Close all open stores

    # return


def plot_spectrum(spectrum: Spectrum, ax):

    # Plot the data
    ax.plot(spectrum.velos, spectrum.input_data, c="red", alpha=0.5)

    # Plot all components
    for component in spectrum.components:
        ax.plot(spectrum.velos, component, c="k", alpha=0.5)

    # Plot the total modelled signal
    ax.plot(spectrum.velos, spectrum.total_model, c="blue")

    # Add a textbox
    textstr = "\n".join(
        (
            f"idx = {spectrum.hpxindex}",
            f"(l, b) = {spectrum.glon:.1f}, {spectrum.glat:.1f}",
            f"# components = {spectrum.n_components}",
        )
    )

    textbox_props = dict(boxstyle="round", facecolor="None", alpha=0.5)

    ax.text(
        0.05,
        0.95,
        textstr,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=textbox_props,
    )

    return ax


def plot_spectra(config: Dict, spectra: np.ndarray, grid_shape: Tuple) -> None:
    # Build figure
    # Set figsize based on grid_shape
    nrows, ncols = grid_shape[0], grid_shape[1]
    figsize = (5 * ncols, 3 * nrows)  # figsize is width X heigth
    fig, axes = plt.subplots(figsize=figsize, nrows=nrows, ncols=ncols)

    # Use a function that plot a spectrum in a single panel
    # for ax, spectrum in zip(axes.flatten(), spectra):
    # plot_spectra()
    plot_iterator = it.starmap(plot_spectrum, zip(spectra, axes.flatten()))
    deque(plot_iterator, maxlen=0)

    # Set labels, legends, ticks
    for ax in axes.flatten():
        # ax.tick_params(labelbottom=False)
        ax.set_xlim(-100, 100)

    # Put all labels on bottom left panel
    axis_lowerleft = axes.flatten()[ncols * (nrows - 1)]
    axis_lowerleft.set_xlabel(r"$v_{\rm LSR}\ [\rm km/s]$")
    axis_lowerleft.set_ylabel(r"$T_{\rm B}\ [\rm K]$")

    # Save to plotdir
    outname = config["paths"]["plotdir"].joinpath(f"sample_spectra.{g_backend}")
    plt.savefig(outname, dpi=300)

    return


@dataclass
class Histogram1DConfig:
    range: Optional[Tuple] = None
    xlabel: Optional[str] = None
    log: bool = False
    bins: int = 50


def plot_histogram_1d(ax, data, hist_config: Histogram1DConfig) -> None:

    ax.hist(data, log=hist_config.log, range=hist_config.range, bins=hist_config.bins)

    # Label and title
    ax.set_xlabel(hist_config.xlabel)
    ax.set_ylabel(r"$\# \rm points$")

    return


def plot_histograms_1d(config: Dict):

    # Build configs
    hist_config_amps = Histogram1DConfig(
        range=[0, 50], xlabel=r"$\rm Peak\, amplitude\ [K]$", log=True
    )

    hist_config_centers = Histogram1DConfig(
        range=[-90, 90], xlabel=r"$v_0\ [\rm km/s]$", log=False
    )

    hist_config_sigmas = Histogram1DConfig(
        range=[0, 15], xlabel=r"$\sigma_v \ [\rm km/s]$", log=False
    )

    hist_configs = (hist_config_amps, hist_config_centers, hist_config_sigmas)

    # Set column names
    column_names = ("peak_amplitude", "center_kms", "sigma_kms")

    # Open GaussDec table
    with tables.open_file(
        config["paths"]["procdir"].joinpath("gaussdec.h5")
    ) as model_store:

        # Model PyTable
        model_table = model_store.root.gaussdec

        # Create fig
        fig, axes = plt.subplots(nrows=3, figsize=(6, 14))

        # Plot all histograms
        hist_iterator = it.starmap(
            plot_histogram_1d,
            zip(
                axes.flatten(),
                (getattr(model_table.cols, n) for n in column_names),
                hist_configs,
            ),
        )
        deque(hist_iterator, maxlen=0)

        # Save plot
        outname = config["paths"]["plotdir"].joinpath(f"histograms_1d.{g_backend}")
        plt.savefig(outname, dpi=300)

        return


def plot_histograms_2d(config: Dict) -> None:
    # Only plotting v0 vs. log(sigma) for now
    with tables.open_file(
        config["paths"]["procdir"].joinpath("gaussdec.h5")
    ) as model_store:

        # Model PyTable
        model_table = model_store.root.gaussdec

        # Create fig
        fig, ax = plt.subplots(figsize=(8, 5))

        center_kms = model_table.cols.center_kms[:]
        fwhm_kms = model_table.cols.sigma_kms[:] * 2.355

    # Make 2D hist
    *_, im = ax.hist2d(
        center_kms,
        np.log10(fwhm_kms),
        range=[[-100, 100], [0.0, 2.2]],
        bins=[80, 50],
        cmap="afmhot_r",
        norm=LogNorm(),
    )

    # Color bar
    cbar = plt.colorbar(im)
    cbar.set_label(r"$\# \rm points$")

    # Save
    outname = config["paths"]["plotdir"].joinpath(f"hist2d_center_sigma.{g_backend}")
    plt.savefig(outname, dpi=300)

    return


def analyze(config):
    # Maps
    make_maps(config)
    plot_maps(config)

    # Spectra
    grid_shape = (6, 4)
    spectra = make_spectra(config, grid_shape=grid_shape)
    plot_spectra(config, spectra, grid_shape=grid_shape)

    # Histograms
    plot_histograms_1d(config)
    plot_histograms_2d(config)


@click.command()
@click.option("--modeldir", help="Model directory.")
@click.option(
    "--backend",
    default="pdf",
    type=click.Choice(["png", "pdf"]),
    help="Either pdf or png",
)
def main(modeldir, backend):
    # Set up logger
    logging.basicConfig(level=logging.INFO, **misc.LOGGING_KW)
    logging.info("Initialized logger")

    # Parse config file
    configfile = Path(modeldir).joinpath("decompose.yaml")
    config = misc.parse_config(configfile)

    # Set plotting backend (pdf or png)
    global g_backend
    g_backend = backend

    analyze(config)
    return


if __name__ == "__main__":
    main()
