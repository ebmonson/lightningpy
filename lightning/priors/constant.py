import numpy as np
from .base import AnalyticPrior

class ConstantPrior(AnalyticPrior):
    '''
    Holding a parameter constant is not really a prior so much as a reduction
    in the dimensionality of the problem, so this is basically a dummy
    prior, which tells the sampler to hold a parameter constant and what its
    value should be.
    '''

    type = 'analytic'
    model_name = 'constant'
    Nparams = 1
    param_names = ['value']
    param_descr = ['Value']
    param_bounds = np.array([-np.inf, np.inf]).reshape(1,2)

    def __init__(self, params):
        super().__init__(params)
        self.val = self.params[0]

    def evaluate(self, x):

        return (self.val == x).astype('int')

    def quantile(self, q):

        return self.val + np.zeros_like(q)

    def sample(self, size):

        return self.val + np.zeros(size)
