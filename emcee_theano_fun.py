import theano.tensor as T
import theano
from theano.ifelse import ifelse
import numpy as np

import emcee

import healpy
import pylab as pl

default_hpars_partial = {
    'mu_cold' : {
        'mu' : 1.7,
        'sigma' : 1.0,
    },
    'sigma_cold' : {
        'x_min' : 1e-1,
        'x_max' : 1e+1,
    },
    'mu_warm' : {
        'mu' : 0.7,
        'sigma' : 1.0,
    },
    'sigma_warm' : {
        'x_min' : 1e-1,
        'x_max' : 1e+1,
    },
    'sigma_model' : {
        'x_min' : 1e-1,
        'x_max' : 1e+1,
    },
}

default_hpars_pooled = {
    'epsilon_cold' : {
        'x_min' : 0.,
        'x_max' : 2.,
    },
    'epsilon_warm' : {
        'x_min' : 0.,
        'x_max' : 2.,
    },
    'offset' : {
        'mu' : 0,
        'sigma' : 3,
    },
    'sigma_model' : {
        'x_min' : 1e-1,
        'x_max' : 1e+1,
    }
}


def t_gauss_lnlike(x, mu, sigma):
    return -0.5 * (x - mu) ** 2 / sigma ** 2 - T.log(sigma)


def t_gumbel_lnlike(x, mu, beta):
    return -1.0 / beta * (x - mu) - T.exp(-1.0 / beta * (x - mu)) - T.log(beta)


def t_uniform_lnlike(x, x_min, x_max):
    return ifelse(x < x_max, ifelse(x > x_min, 0.0, -np.inf), -np.inf)


def uniform_rand(x_min, x_max, n_values=1):
    return np.random.uniform(x_min, x_max, n_values)


def gauss_rand(mu, sigma, n_values=1):
    return np.random.normal(mu, sigma, n_values)


def compile_partial_pooling_lnlike(hpars):

    # Set up tensor variables
    p = T.vector()
    tau, cold, warm = T.vectors(3)
    n_values = tau.size

    # split parameters
    mu_cold, sigma_cold, mu_warm, sigma_warm, sigma_model = p[0], p[1], p[2], p[3], p[4]
    epsilon_cold_i = p[5 + 0 * n_values : 5 + 1 * n_values]
    epsilon_warm_i = p[5 + 1 * n_values : 5 + 2 * n_values]

    # Priors on partial-pooling distributions
    # and model error
    mu_cold_lnprior = t_gauss_lnlike(mu_cold, **hpars['mu_cold'])
    sigma_cold_lnprior = t_uniform_lnlike(sigma_cold, **hpars['sigma_cold'])
    mu_warm_lnprior = t_gauss_lnlike(mu_warm, **hpars['mu_warm'])
    sigma_warm_lnprior = t_uniform_lnlike(sigma_warm, **hpars['sigma_warm'])
    sigma_model_lnprior = t_uniform_lnlike(sigma_model, **hpars['sigma_model'])

    # Priors on emissivities
    epsilon_cold_lnprior = T.sum(t_gauss_lnlike(epsilon_cold_i, mu_cold, sigma_cold))
    epsilon_warm_lnprior = T.sum(t_gauss_lnlike(epsilon_warm_i, mu_warm, sigma_warm))

    mu_model = epsilon_cold_i * cold + epsilon_warm_i * warm

    model = T.sum(t_gauss_lnlike(tau, mu_model, sigma_model))

    likelihood = sum([model,
        epsilon_warm_lnprior,epsilon_cold_lnprior,
        mu_cold_lnprior, sigma_cold_lnprior,
        mu_warm_lnprior, sigma_warm_lnprior,
        sigma_model_lnprior])

    likelihood_function = theano.function(inputs=[p, tau, cold, warm], outputs=likelihood)

    return likelihood_function


def compile_pooled_lnlike(hpars):

    p, tau, cold, warm = T.vectors(4)

    epsilon_cold, epsilon_warm, offset, sigma_model = p[0], p[1], p[2], p[3]

    epsilon_cold_lnprior = t_uniform_lnlike(epsilon_cold, **hpars['epsilon_cold'])
    epsilon_warm_lnprior = t_uniform_lnlike(epsilon_warm, **hpars['epsilon_warm'])
    offset_lnprior = t_gauss_lnlike(offset, **hpars['offset'])
    sigma_model_lnprior = t_uniform_lnlike(sigma_model, **hpars['sigma_model'])

    mu_model = epsilon_cold * cold + epsilon_warm * warm + offset
    model = T.sum(t_gumbel_lnlike(tau, mu_model, sigma_model))

    likelihood = sum([model,
        epsilon_cold_lnprior,
        epsilon_warm_lnprior,
        offset_lnprior,
        sigma_model_lnprior])

    likelihood_function = theano.function(inputs=[p, tau, cold, warm], outputs=likelihood)

    return likelihood_function


def pooled_lnlike(tau, cold, warm, hpars):

    n_values = tau.size
    n_dim = 4

    likelihood = compile_pooled_lnlike(hpars)

    def lnfun(p):
        l = likelihood(p, tau, cold, warm)
        return l if np.isfinite(l) else -np.inf

    def dfun():
        return np.concatenate([
            gauss_rand(**hpars['epsilon_cold']),
            gauss_rand(**hpars['epsilon_warm']),
            gauss_rand(**hpars['offset']),
            uniform_rand(**hpars['sigma_model'])])

    return lnfun, dfun, n_dim, n_values


def partial_pooling_lnlike(tau, cold, warm, hpars):

    n_values = tau.size
    n_dim = 5 + 2 * n_values

    likelihood = compile_partial_pooling_lnlike(hpars)

    def lnfun(p):
        l = likelihood(p, tau, cold, warm)
        return l if np.isfinite(l) else -np.inf

    def dfun():
        return np.concatenate([
            gauss_rand(**hpars['mu_cold']),
            uniform_rand(**hpars['sigma_cold']),
            gauss_rand(**hpars['mu_warm']),
            uniform_rand(**hpars['sigma_warm']),
            uniform_rand(**hpars['sigma_model']),
            gauss_rand(n_values=n_values, **hpars['mu_cold']),
            gauss_rand(n_values=n_values, **hpars['mu_warm'])])

    return lnfun, dfun, n_dim, n_values


def get_data(nested=False):

    nhi_scale = 1.82e18 * 1.288 / 1.0e20
    tau_scale = 1.0e6

    cold = np.load('../data/cold.npy') * nhi_scale
    warm = np.load('../data/warm.npy') * nhi_scale
    tau  = np.load('../data/tau.npy') * tau_scale

    mask = (cold > 0) | (warm > 0)

    if nested:
        cold = healpy.reorder(cold, r2n=True)
        warm = healpy.reorder(warm, r2n=True)
        tau = healpy.reorder(tau, r2n=True)
        mask = healpy.reorder(mask, r2n=True)

    return cold, warm, tau, mask


def make_sampler(tau, cold, warm, hpars, n_walkers=500, lntype=pooled_lnlike):

    likelihood_fn, init_fn, n_dim, n_values = lntype(tau, cold, warm, hpars)
    
    n_walkers = max(n_dim * 2, n_walkers)
    p0 = [init_fn() for i in xrange(n_walkers)]

    sampler = emcee.EnsembleSampler(n_walkers, n_dim, likelihood_fn)
    
    return sampler, p0


def superpixel_mask(nside, steps):

    idx = np.arange(healpy.nside2npix(nside))
    m = np.right_shift(idx, 2 * steps)

    return m


if __name__ == '__main__':

    cold, warm, tau, mask = get_data()
    sp_mask = superpixel_mask(1024, 5)
    pixel = 40

    tau_s  = tau[mask & (sp_mask == pixel)]
    cold_s = cold[mask & (sp_mask == pixel)]
    warm_s = warm[mask & (sp_mask == pixel)]

    sampler, p0 = make_sampler(tau_s, cold_s, warm_s, default_hpars_pooled)
    sampler.run_mcmc(p0, 500)

    for i in xrange(4):
        pl.figure()
        pl.subplot(211)
        pl.plot(sampler.chain[:,:,i].T, color='k', alpha=0.1)
        pl.subplot(212)
        pl.hist(sampler.chain[:,200:,i].ravel(), bins=100)

    pl.show()