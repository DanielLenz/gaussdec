import theano
import theano.tensor as T

import numpy as np
from scipy.interpolate import interp1d

def make_multi_gaussian_model():

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
    weighted_residual = residual / (20. + y)
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


def initial_centers_pdf(coordinates, values, threshold=0.1):

    cdf = np.cumsum(values > threshold).astype(float)
    cdf /= cdf[-1]

    cdf_interp = interp1d(cdf, coordinates, fill_value=0, bounds_error=False)

    def get_centers(n_centers):
        return cdf_interp(np.random.uniform(0,1,n_centers))

    return get_centers


default_p = {
    'min_components' : 1,
    'max_components' : 10,
    'iterations' : 10,
    'int_low' : 5e18 / 1.82e18 / 1.28,
    'int_high' : 1e21 / 1.82e18 / 1.28,
    'sigma_low' : np.sqrt(50 / 21.85) / 1.28 / 2.35,
    'sigma_high' : np.sqrt(10000 / 21.85) / 1.28 / 2.35,
    'pdf_threshold' : 0.1,
    'pdf_kernel' : 0, 
}


def fit_spectrum(y, p):

