import numpy as np
import healpy as hp
from sys import path

# paths
BASEPATH = '/users/dlenz/projects/'
PROJECTPATH = '/users/dlenz/projects/gaussdec/'

# healpy
NSIDE = 1024
NPIX = hp.nside2npix(NSIDE)

# plotting keyword args
IMKW = dict(origin='lower', interpolation='None', cmap='YlGnBu_r')
PLKW = dict(drawstyle='steps', c='k')
CONTLEVELS = 10**np.arange(1., 5., 0.75)


# EBHISchannel*Kelvin to column density
cK2nh = 1.822e18 * 1.288

# figsize in pixels
FIGSIZE_1COL = [3.5433, 2.1898]
FIGSIZE_2COL = [7.2440, 4.4770]
