import numpy as np

from ..attenuation import TabulatedAtten

class Tbabs(TabulatedAtten):
    '''Tubingen-Boulder absorption model.

    Includes cross sections from gas phase ISM, grains, and molecular hydrogen.
    Atomic abundances are fixed to the default.

    Parameters
    ----------
    wave : np.ndarray, (Nwave,), float
        Rest frame wavelength grid to evaluate the model on.
    path_to_models : str
        Path to lightning models. Not actually used in normal circumstances.

    References
    ----------
    - `<https://heasarc.gsfc.nasa.gov/xanadu/xspec/manual/XSmodelTbabs.html>`_
    - `<https://ui.adsabs.harvard.edu/abs/2000ApJ...542..914W/abstract>`_

    '''

    type = 'tabulated'
    model_name = 'tbabs'
    Nparams = 1
    param_names = ['NH']
    param_descr = ['Hydrogen column density']
    param_names_fncy = [r'$N_H$']
    param_bounds = np.array([0, 1e5]).reshape(1,2)
    path = 'xray/abs/tbabs.txt'

    def evaluate(self, params):
        '''Evaluate the absorption as a function of wavelength for the given parameters.

        Parameters
        ----------
        params : np.ndarray, (Nmodels, 1) or (1,)
            Values for NH.

        Returns
        -------
        expminustau : (Nmodels, Nwave)

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
    '''Photo-electric absorption model.

    Abundances are fixed to the default.

    Parameters
    ----------
    wave : np.ndarray, (Nwave,), float
        Rest frame wavelength grid to evaluate the model on.
    path_to_models : str
        Path to lightning models. Not actually used in normal circumstances.

    References
    ----------
    - `<https://heasarc.gsfc.nasa.gov/xanadu/xspec/manual/node264.html>`_

    '''

    type = 'tabulated'
    model_name = 'phabs'
    Nparams = 1
    param_names = ['NH']
    param_descr = ['Hydrogen column density']
    param_names_fncy = [r'$N_H$']
    param_bounds = np.array([0, 1e5]).reshape(1,2)
    path = 'xray/abs/phabs.txt'

    def evaluate(self, params):
        '''Evaluate the absorption as a function of wavelength for the given parameters.

        Parameters
        ----------
        params : np.ndarray, (Nmodels, 1) or (1,)
            Values for NH.

        Returns
        -------
        expminustau : (Nmodels, Nwave)

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
