======
DESIGN
======

Requirements
============

- Gaussian decomposition of spectral data in the HEALPix format
- Easy iteration cycle, from parameters updates to all analysis results
- Solid unit and integration testing
- 

Architecture
============

Inputs
------

- Parameters passed as config files (decompose.yaml?)
    - Input file
    - output directory (creates raw/, processed/, analysis/)
    - Parameters for the fit
        - ncomps
        - velocity range
        - All priors (format for priors?)
        - Parameters for submission (number of splits, for instance)
    - 

Decomposition
-------------
- Core: Fit one pixel at a time
- Use imap to distribute to all CPUs
- Have a dedicated submit/ part that slices/prepares the data, distributes across nodes
- Can we build a pipeline that calls the postprocessing automatically?

Postprocess
-----------
- Maps
    - If necessary, merge data into one large table
    - Build NHI in for the given velocity range

Analysis
--------

- Maps
    - Build NHI model and residual
    - LVC/IVC/HVC gas (for neg/pos velos)
    - Cold and warm NHI (also for different LVC/IVC/HVC?)
    - n components
    - Amp-weighted line width
    - Amp-weighted v0
- Histograms
    - 1D and 2D
    - All combinations for 2D, follow Kalberla & Haud (2018) limits/scaling
- Sample spectra
    - On a given page, show a large variety of spectra
    - Randomize each time
    - Sample different lon/lat/densities
    - Show all model components + total + input
    - Text box with model parameters (ncomps)
    - Vertical lines at components
    - Separate ax for residual

