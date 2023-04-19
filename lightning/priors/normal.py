import numpy as np
from scipy.special import erfinv
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
        '''
        Return an array with the same shape as ``x`` that is
        equal to::

        p = 1 / [sigma * sqrt(2 * pi)] * exp[-1 * (x - mu)**2 / sigma**2)]

        '''

        mu = self.params[0]
        sigma = self.params[1]

        p = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-1 * ((x - mu) / sigma)**2)

        return p

    def quantile(self, q):
        '''
        Return an array with the same shape as ``q`` that is
        equal to::

        x = mu + sigma * sqrt(2) * erfinv(2 * q - 1)

        '''

        mu = self.params[0]
        sigma = self.params[1]

        x = mu + sigma * np.sqrt(2) * erfinv(2 * q - 1)

        return x
        
