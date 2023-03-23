'''
    priors.py

    Classes for priors on parameters. Priors are
    currently on a per-parameter basis, but this
    implementation is general enough to make it so
    that we could have priors tying parameters together.
    BUT, that would mean changing how the priors are
    incorporated into the log probablility in the main
    class.
'''

import numpy as np
from scipy.interpolate import interp1d

###############################
# Priors with a functional form
###############################
class AnalyticPrior():

    type = 'analytic'
    model_name = None
    Nparams = None
    param_names = ['None']
    param_descr = ['None']
    param_bounds = np.array([None, None]).reshape(1,2)

    def __init__(self, params):
        '''
            Need only be overwritten if there's specific requirements for the prior parameters
            (i.e. b > a for the uniform prior).
        '''

        if self.Nparams is not None:
            assert (self.Nparams == len(params)), "Number of parameters must match the number (%d) expected by this model (%s)" % (self.Nparams, self.model_name)

            for i in np.arange(self.Nparams):
                if np.any((params < self.param_bounds[:,0]) | (params > self.param_bounds[:,1])):
                    raise ValueError('Supplied prior parameter are out of bounds for this model (%s).' % (self.model_name))

        self.params = params

    def __call__(self, x):
        return self.evaluate(x)

    def evaluate(self, x):
        '''
            This function must be overwritten by each specific prior.
        '''

        return np.zeros_like(x)

class UniformPrior(AnalyticPrior):

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
        b = self.params[1]
        a = self.params[0]

        p = np.zeros_like(x)
        p[(x >= a) & (x < b)] = 1 / (b - a)

        return p

class NormalPrior(AnalyticPrior):

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

#################################
# Priors without a function form,
# interpolated from a table.
#################################

class TabulatedPrior():

    type = 'tabulated'
    model_name = 'tabulated'
    Nparams = 0
    param_names = ['None']
    param_descr = ['None']
    params_bounds = np.array([None, None]).reshape(1,2)

    def __init__(self, x, y, **kwargs):
        '''
            Keywords are passed on to scipy.interpolate.interp1d
        '''

        assert (len(x) == len(y)), "Number of probability values (%d) must match number of independent variable values (%d)." % (len(y), len(x))

        self.callable = interp1d(x, y, **kwargs)

    def evaluate(self, x):

        return self.callable(x)

    def __call__(self, x):

        return self.evaluate(x)
