import numpy as np
from .base import AnalyticPrior

class UniformPrior(AnalyticPrior):
    r'''Uniform prior. Parameters are lower bound a and upper bound b, in that order.

    PDF:

    .. math::

        p(x) = \begin{cases}
                    \frac {1} {b - a} &, ~x \in [a, b) \\
                    0 &,~{\rm otherwise}
                \end{cases}

    Quantile function:

    .. math::

        x(q) = q (b - a) + a
        

    '''

    type = 'analytic'
    model_name = 'uniform'
    Nparams = 2
    param_names = ['lo_bound', 'hi_bound']
    param_descr = ['Lower bound', 'Upper bound']
    param_bounds = np.array([[-np.inf, np.inf],
                             [-np.inf,np.inf]])

    def __init__(self, params):

        if self.Nparams is not None:
            assert (self.Nparams == len(params)), "Number of parameters must match the number (%d) expected by this model (%s)" % (self.Nparams, self.model_name)

            for i in np.arange(self.Nparams):
                if np.any((params < self.param_bounds[:,0]) | (params > self.param_bounds[:,1])):
                    raise ValueError('Supplied prior parameter are out of bounds for this model (%s).' % (self.model_name))

        assert (params[1] > params[0]), "For the %s model, '%s' must be greater than '%s'" % (self.model_name, self.param_names[1], self.param_names[0])

        self.params = params

    def evaluate(self, x):
        '''
        Return an array with the same shape as x that's equal to ``1 / (b - a)``
        wherever x is in [a,b) and 0 elsewhere.
        '''

        b = self.params[1]
        a = self.params[0]

        p = np.zeros_like(x)
        p[(x >= a) & (x < b)] = 1 / (b - a)

        return p

    def quantile(self, q):
        '''
        Return an array with the same shape as q that's equal to ``q * (b - a) + a``.
        '''

        b = self.params[1]
        a = self.params[0]

        x = q * (b - a) + a

        return x
