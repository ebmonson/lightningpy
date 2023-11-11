'''
    Single exponential star formation history.
'''

import numpy as np
from .base import FunctionalSFH

class SingleExponentialSFH(FunctionalSFH):
    '''
    Exponentially decaying burst of star formation:

    sfr(t) = {A exp[(t - tburst) / tau] / k, t <= tburst
             {0, t > tburst

    where k = tau * [1 - exp(-tburst / tau)]
    '''

    type = 'functional'
    model_name = 'Single-Exponential'
    Nparams = 3
    param_names = ['sexp_norm', 'sexp_tburst', 'sexp_tau']
    param_descr = ['Integral of the SFH',
                   'Burst age',
                   'Decay constant/e-folding time of the SFH']
    param_names_fncy = [r'$\rm Norm_{SFH}$', r'$t_{\rm burst}$', r'$\tau_{\rm SFH}$']
    param_bounds = np.array([[0.0, np.inf],
                             [1, 1.4e10],
                             [1, 1.4e10]])


    def __init__(self, age):

        self.age = age

    def evaluate(self, params):
        '''
            Returns the SFR as a function of time.
        '''

        # Check that the model is defined for the given parameters
        ob = self._check_bounds(params)
        #ob = (np.any(params < self.param_bounds[:,0][None,:], axis=1) | np.any(params > self.param_bounds[:,1][None,:], axis=1))
        if (np.any(ob)):
            # Failing loudly vs quietly (and returning infs or nans or something)
            # for the out-of-bounds models is TBD.
            raise ValueError('Given parameters are out of bounds for this model (%s).' % (self.model_name))

        if len(params.shape) == 1:
            params = params.reshape(1,-1)

        if self.Nparams is not None:
            assert (self.Nparams == params.shape[1]), "Number of parameters must match the number (%d) expected by this model (%s)" % (self.Nparams, self.model_name)
        Nmodels = params.shape[0]

        A = params[:,0]
        tburst = params[:,1]
        tau = params[:,2]

        k = tau * (1 - np.exp(-tburst / tau))

        sfrt = A[:,None] / k[:,None] * np.exp((self.age[None,:] - tburst[:,None]) / tau[:,None])

        noSF = self.age[None,:] > tburst[:,None]

        sfrt[noSF] = 0.0

        if Nmodels == 1:
            sfrt = sfrt.flatten()

        return sfrt
