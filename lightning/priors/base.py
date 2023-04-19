import numpy as np
from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d

class AnalyticPrior():
    '''Base class for priors with a functional form.

    Need only be overwritten if there's specific requirements for the prior parameters
    (i.e. b > a for the uniform prior).
    '''

    type = 'analytic'
    model_name = None
    Nparams = None
    param_names = ['None']
    param_descr = ['None']
    param_bounds = np.array([None, None]).reshape(1,2)

    def __init__(self, params):

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
        This function must be overwritten by each specific prior,
        implementing the PDF.
        '''

        return np.zeros_like(x)

    def quantile(self, q):
        '''
        This function must be overwritten by each specific prior,
        implementing the quantile function/PPF (the inverse of the
        CDF).
        '''

        return np.zeros_like(x)

class TabulatedPrior():
    '''Base class for tabulated priors.

    Keywords are passed on to scipy.interpolate.interp1d.
    '''

    type = 'tabulated'
    model_name = 'tabulated'
    Nparams = 0
    param_names = ['None']
    param_descr = ['None']
    params_bounds = np.array([None, None]).reshape(1,2)

    def __init__(self, x, y, **kwargs):

        assert (len(x) == len(y)), "Number of probability values (%d) must match number of independent variable values (%d)." % (len(y), len(x))

        self.callable = interp1d(x, y, **kwargs)
        cdf = cumtrapz(y, x, initial=0)
        self.inverse = interp1d(cdf, x, **kwargs)

    def evaluate(self, x):
        '''
        Evaluate the scipy.interpolate.interp1d object representing the PDF
        on ``x``.
        '''

        return self.callable(x)

    def quantile(self, q):
        '''
        Evaluate the scipy.interpolate.interp1d object representing the
        quantile function on ``q``.
        '''

        return self.inverse(q)

    def __call__(self, x):

        return self.evaluate(x)
