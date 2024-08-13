'''
    Delayed exponential star formation history.
'''

import numpy as np
from .base import FunctionalSFH

class DelayedExponentialSFH(FunctionalSFH):
    r'''Delayed exponential burst of star formation

    .. math::
        \psi(t) = A (t / \tau) \exp(-t / \tau)

    '''

    type = 'functional'
    model_name = 'Delayed-Exponential'
    Nparams = 2
    param_names = ['dexp_norm', 'dexp_tau']
    param_descr = ['Normalization of the SFH: int(sfr(t), 0, inf) = dexp_norm * dexp_tau',
                   'Time delay / decay constant of the SFH']
    param_names_fncy = [r'$\rm Norm_{SFH}$', r'$\tau_{\rm SFH}$']
    param_bounds = np.array([[0.0, np.inf],
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
        tau = params[:,1]

        sfrt = A[:,None] * (self.age[None,:] / tau[:,None]) * np.exp(-1 * self.age[None,:] / tau[:,None])

        if Nmodels == 1:
            sfrt = sfrt.flatten()

        return sfrt
