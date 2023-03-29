import numpy as np
from .base import AnalyticPrior

class NormalPrior(AnalyticPrior):
    '''
        Normal prior.
    '''

    type = 'analytic'
    model_name = 'normal'
    Nparams = 2
    param_names = ['mu', 'sigma']
    param_descr = ['Mean', 'Sigma']
    param_bounds = np.array([[-np.inf, np.inf],
                             [0, np.inf]])

    def evaluate(self, x):

        mu = self.params[0]
        sigma = self.params[1]

        p = 1 / np.sqrt(2 * np.pi * sigma) * np.exp(-1 * ((x - mu) / sigma)**2)

        return p
