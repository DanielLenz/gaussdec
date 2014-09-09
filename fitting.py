import theano
import theano.tensor as T

import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize

def make_multi_gaussian_model(tsys=20):

    x = T.vector('x')
    y = T.vector('y')
    sx = x.dimshuffle(0,'x')

    parameters = T.vector('parameters')
    amplitudes = parameters[0::3].dimshuffle('x',0)
    centers = parameters[1::3].dimshuffle('x',0)
    dispersions = parameters[2::3].dimshuffle('x',0)

    components = amplitudes / T.sqrt(2 * np.pi) / dispersions * T.exp(-0.5 * (sx - centers) * (sx - centers) / dispersions / dispersions)
    multigauss = T.sum(components, 1)
    residual = y - multigauss
    weighted_residual = residual / (tsys + y)
    objective = T.sum(weighted_residual * weighted_residual)
    bic = y.shape[0] * T.log(objective / (y.shape[0] - 1)) + parameters.shape[0] * T.log(y.shape[0])
    jacobian = theano.gradient.jacobian(objective, parameters)

    f_jacobian = theano.function(
        inputs=[parameters, x, y],
        outputs=jacobian)

    f_model = theano.function(
        inputs=[parameters, x],
        outputs=[components, multigauss])

    f_residual = theano.function(
        inputs=[parameters, x, y],
        outputs=residual)

    f_objective = theano.function(
        inputs=[parameters, x, y],
        outputs=objective)

    f_stats = theano.function(
        inputs=[parameters, x, y],
        outputs=[bic])

    return f_model, f_residual, f_objective, f_jacobian, f_stats


def initial_centers_pdf(coordinates, values, threshold=0.1, kernel=0., trim=10):

    trim_mask = np.arange(values.shape[0])
    trim_mask[:trim] = 0
    trim_mask[-trim:] = 0
    
    sm_values = gaussian_filter1d(values * trim_mask, kernel, mode="constant", cval=0)

    cdf = np.cumsum(sm_values > threshold).astype(float)
    cdf /= cdf[-1]

    cdf_interp = interp1d(cdf, coordinates, fill_value=0, bounds_error=False)

    def get_centers(n_centers):
        return cdf_interp(np.random.uniform(0, 1, n_centers))

    return get_centers


default_p = {
    'min_components' : 1,
    'max_components' : 10,
    'iterations' : 10,
    'int_low' : 5e18 / 1.82e18 / 1.28,
    'int_high' : 1e21 / 1.82e18 / 1.28,
    'sigma_low' : np.sqrt(50 / 21.85) / 1.28 / 2.35,
    'sigma_high' : np.sqrt(40000 / 21.85) / 1.28 / 2.35,
    'pdf_threshold' : 0.09,
    'pdf_kernel' : 3.32, 
    'fit_method' : 'l-bfgs-b',
    'trim' : 10,
}


def fit_spectrum(y, objective, jacobian, stats, p):

    x = np.arange(y.shape[0])

    initial_centers = initial_centers_pdf(x, y,
        threshold=p['pdf_threshold'],
        kernel=p['pdf_kernel'],
        trim=p['trim'])

    component_trials = range(p['min_components'], p['max_components'] + 1) * p['iterations']

    def trials():

        for n_components in component_trials:

            x0 = np.array([p['int_low'] * 5, 0, p['sigma_low']] * n_components)
            x0[1::3] = initial_centers(n_components)

            bounds = [
                (p['int_low'], p['int_high']),
                (None, None),
                (p['sigma_low'], p['sigma_high']),
            ] * n_components

            result = minimize(objective, x0,
                args=(x, y),
                jac=jacobian,
                bounds=bounds,
                method=p['fit_method'])

            yield result.x, stats(result.x, x, y)

    trial_results = sorted(list(trials()), key=lambda k:k[1][0])
    t = trial_results[0]
    
    result_keys = ['parameters', 'stats']
    result_values = [t[0].tolist(), map(float, t[1])]
    return {k : v for k,v in zip(result_keys, result_values)}
