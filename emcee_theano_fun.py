import theano.tensor as T
import theano
from theano.ifelse import ifelse
import numpy as np

import emcee

import healpy

def compile_likelihood(e_mu, e_sigma, z_mu, z_sigma, s_min, s_max):

    # vector for emcee sampling
    # 0 : e_cold
    # 1 : e_warm
    # 2 : z
    # 3 : s
    
    p = T.vector()
    e_cold, e_warm, z, s = p[0], p[1], p[2], p[3]
    tau, cold, warm = T.vectors(3)

    e_cold_lnprior = -0.5 * (e_cold - e_mu) ** 2 / e_sigma ** 2 - T.log(e_sigma)
    e_warm_lnprior = -0.5 * (e_warm - e_mu) ** 2 / e_sigma ** 2 - T.log(e_sigma)
    z_lnprior      = -0.5 * (z - z_mu) ** 2 / z_sigma ** 2 - T.log(z_sigma)
    s_lnprior      = ifelse(s < s_max, ifelse(s > s_min, 0.0, -np.inf), -np.inf)

    model_mean = e_cold * cold + e_warm * warm + z
    model_likelihood = T.sum(-0.5 * abs(tau - model_mean) / s - T.log(s))

    likelihood = model_likelihood + e_cold_lnprior + e_warm_lnprior + z_lnprior + s_lnprior

    likelihood_function = theano.function(inputs=[p, tau, cold, warm], outputs=likelihood)

    return likelihood_function


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


def sample():

    cold, warm, tau, mask = get_data(True)

    power = 8
    indices_lowres = np.arange(0, healpy.nside2npix(4))
    i = 1

    low, high = i * 4 ** power, (i + 1) * 4 ** power

    cold_sample = cold[low:high][mask[low:high]]
    warm_sample = warm[low:high][mask[low:high]]
    tau_sample = tau[low:high][mask[low:high]]
    
    e_mu, e_sigma = 1, 10
    z_mu, z_sigma = 0, 3
    s_min, s_max = 1e-2, 100
    
    likelihood = compile_likelihood(e_mu, e_sigma, z_mu, z_sigma, s_min, s_max)

    def dummy_likelihood(p):
        l = likelihood(p, tau_sample, cold_sample, warm_sample)
        return l if np.isfinite(l) else -np.inf

    def draw_start():
        return [
            np.random.normal(e_mu, e_sigma),
            np.random.normal(e_mu, e_sigma),
            np.random.laplace(z_mu, z_sigma),
            np.random.uniform(s_min, s_max)
        ]

    n_dim, n_walkers = 4, 100
    p0 = [draw_start() for i in xrange(n_walkers)]

    sampler = emcee.EnsembleSampler(n_walkers, n_dim, dummy_likelihood)
    sampler.run_mcmc(p0, 500)

    import matplotlib.pyplot as pl

    for i in xrange(n_dim):
        pl.figure()
        pl.subplot(211)
        pl.hist(sampler.chain[:, 200:, i].ravel(), 1000, color="k", histtype="step")
        pl.subplot(212)
        pl.plot(sampler.chain[:,:,i].T, color="k", alpha=0.25)
        pl.title("Dimension {0:d}".format(i))

    posterior_means = [np.mean(sampler.chain[:, 200:, i]) for i in xrange(n_dim)]

    model_sky = np.zeros_like(cold)
    comp_tau = np.zeros_like(tau)
    comp_tau[low:high] = tau[low:high]
    model_sky[low:high] = posterior_means[0] * cold[low:high] + posterior_means[1] * warm[low:high] + posterior_means[2]

    healpy.mollzoom(model_sky, nest=True)
    healpy.mollzoom(model_sky - comp_tau, nest=True)
    pl.show()


if __name__ == '__main__':
    sample()
