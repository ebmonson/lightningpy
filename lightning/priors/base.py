import numpy as np
from scipy.integrate import cumulative_trapezoid as cumtrapz
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

    def sample(self, size, rng=None, seed=None):
        '''Sample from the prior.

        Parameters
        ----------
        size : int
            Number of samples to draw
        rng : numpy.random.Generator
            Numpy object for random number generation;
            see ``numpy.random.default_rng()``
        seed : int
            Seed for random number generation. If you pass
            a pre-constructed generator this is ignored.

        Returns
        -------
        samples : numpy array
            Random samples
        '''

        if rng is None: rng = np.random.default_rng(seed)

        q = rng.uniform(size=size)

        return self.quantile(q)

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

    def sample(self, size, rng=None, seed=None):
        '''Sample from the prior.

        Parameters
        ----------
        size : int
            Number of samples to draw
        rng : numpy.random.Generator
            Numpy object for random number generation;
            see ``numpy.random.default_rng()``
        seed : int
            Seed for random number generation. If you pass
            a pre-constructed generator this is ignored.

        Returns
        -------
        samples : numpy array
            Random samples
        '''

        if rng is None: rng = np.random.default_rng(seed)

        q = rng.uniform(size=size)

        return self.quantile(q)

    def __call__(self, x):

        return self.evaluate(x)
