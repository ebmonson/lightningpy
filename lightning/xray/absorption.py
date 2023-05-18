import numpy as np

from ..attenuation import TabulatedAtten

class Tbabs(TabulatedAtten):

    type = 'tabulated'
    model_name = 'tbabs'
    Nparams = 1
    param_names = ['NH']
    param_descr = ['Hydrogen column density']
    param_bounds = np.array([0, 1e5]).reshape(1,2)
    path = 'xray/abs/tbabs.txt'

    def evaluate(self, params):
        '''
            This method should return e^(-tau) at each
            wavelength.

            It must be overwritten by each specific attenuation model,
            and it should returnan (Nmodels, Nwave) array.
        '''

        params = np.array(params)

        #print('Before: ', params.shape)

        if len(params.shape) <= 1:
            params = params.reshape(-1, 1)

        #print('After: ', params.shape)

        ob = self._check_bounds(params)
        if (np.any(ob)):
            raise ValueError('Given parameters are out of bounds for this model (%s).' % (self.model_name))

        if self.Nparams is not None:
            assert (self.Nparams == params.shape[1]), "Number of parameters must match the number (%d) expected by this model (%s)" % (self.Nparams, self.model_name)
        Nmodels = params.shape[0]

        # exp(-tau) = exp(-NH * sigma) = exp[-1 * (NH / 1e20) * 1e20 * sigma]
        #           = exp[-1e20 * sigma] ** (NH / 1e20)
        expminustau = self.expminustau_normed[None,:] ** params

        if Nmodels == 1:
            expminustau = expminustau.flatten()

        return expminustau

class Phabs(TabulatedAtten):

    type = 'tabulated'
    model_name = 'phabs'
    Nparams = 1
    param_names = ['NH']
    param_descr = ['Hydrogen column density']
    param_bounds = np.array([0, 1e5]).reshape(1,2)
    path = 'xray/abs/phabs.txt'

    def evaluate(self, params):
        '''
            This method should return e^(-tau) at each
            wavelength.

            It must be overwritten by each specific attenuation model,
            and it should returnan (Nmodels, Nwave) array.
        '''

        params = np.array(params)

        if len(params.shape) <= 1:
            params = params.reshape(-1,1)

        ob = self._check_bounds(params)
        if (np.any(ob)):
            raise ValueError('Given parameters are out of bounds for this model (%s).' % (self.model_name))

        if self.Nparams is not None:
            assert (self.Nparams == params.shape[1]), "Number of parameters must match the number (%d) expected by this model (%s)" % (self.Nparams, self.model_name)
        Nmodels = params.shape[0]

        # exp(-tau) = exp(-NH * sigma) = exp[-1 * (NH / 1e20) * 1e20 * sigma]
        #           = exp[-1e20 * sigma] ** (NH / 1e20)
        expminustau = self.expminustau_normed[None,:] ** params

        if Nmodels == 1:
            expminustau = expminustau.flatten()

        return expminustau
