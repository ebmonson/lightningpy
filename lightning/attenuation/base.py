from pathlib import Path
import numpy as np
from scipy.interpolate import interp1d

#################################
# Dust Attenuation
#################################
class AnalyticAtten:
    '''
    Base class for analytic (i.e., not tabulated)
    attenuation curves.
    '''

    type = 'analytic'
    model_name = None
    Nparams = None
    param_names = ['None']
    param_descr = ['None']
    param_bounds = np.array([-np.inf, np.inf]).reshape(1,2)

    def __init__(self, wave):

        self.wave = wave
        self.Nwave = len(self.wave)

    def _check_bounds(self, params):
        '''
        Check that the parameters are within the ranges where the model is
        meaningful and defined. Return the indices where the model
        is out of bounds.
        '''

        if len(params.shape) == 1:
            params = params.reshape(1,-1)

        if self.Nparams is not None:
            assert (self.Nparams == params.shape[1]), "Number of parameters must match the number (%d) expected by this model (%s)" % (self.Nparams, self.model_name)
        Nmodels = params.shape[0]

        ob_idcs = (params < self.param_bounds[:,0][None,:]) | (params > self.param_bounds[:,1][None,:])

        return ob_idcs


    def evaluate(self, params):
        '''
        This method should return e^(-tau) at each
        wavelength.

        It must be overwritten by each specific attenuation model,
        and it should returnan (Nmodels, Nwave) array.
        '''

        if len(params.shape) == 1:
            params = params.reshape(1,-1)

        ob = self._check_bounds(params)
        if (np.any(ob)):
            raise ValueError('Given parameters are out of bounds for this model (%s).' % (self.model_name))

        if self.Nparams is not None:
            assert (self.Nparams == params.shape[1]), "Number of parameters must match the number (%d) expected by this model (%s)" % (self.Nparams, self.model_name)
        Nmodels = params.shape[0]

        expminustau = np.zeros((Nmodels, len(self.wave)))

        if Nmodels == 1:
            expminustau = expminustau.flatten()

        return expminustau

class TabulatedAtten:
    '''
Base class for tabulated attenuation curves.
    '''

    type = 'tabulated'
    model_name = None
    Nparams = None
    param_names = ['None']
    param_descr = ['None']
    param_bounds = np.array([-np.inf, np.inf]).reshape(1,2)
    path = 'None'

    def __init__(self, wave=None, path_to_models=None):

        if (path_to_models is None):
            self.path_to_models = str(Path(__file__).parent.resolve()) + '/../models/'
        else:
            self.path_to_models = path_to_models
            if(self.path_to_models[-1] != '/'): self.path_to_models = self.path_to_models + '/'

        if (self.path != 'None'):
            arr = np.loadtxt(self.path_to_models + self.path, dtype='float')
            wave_src = arr[:,0]
            expminustau_src = arr[:,-1]

        if (wave is not None):
            # Extrapolation turned on, but all it does is propagate the edge
            # values at a constant level.
            finterp = interp1d(wave_src, expminustau_src, bounds_error=False,
                               fill_value=(expminustau_src[0], expminustau_src[-1]))

            self.wave = wave
            self.expminustau_normed = finterp(wave)

        else:

            self.wave = wave_src
            self.expminustau_normed = expminustau_src

        self.Nwave = len(self.wave)

    def _check_bounds(self, params):
        '''
        Check that the parameters are within the ranges where the model is
        meaningful and defined. Return the indices where the model
        is out of bounds.
        '''

        if len(params.shape) == 1:
            params = params.reshape(1,-1)

        if self.Nparams is not None:
            assert (self.Nparams == params.shape[1]), "Number of parameters must match the number (%d) expected by this model (%s)" % (self.Nparams, self.model_name)
        Nmodels = params.shape[0]

        ob_idcs = (params < self.param_bounds[:,0][None,:]) | (params > self.param_bounds[:,1][None,:])

        return ob_idcs

    def evaluate(self, params):
        '''
        This method should return e^(-tau) at each
        wavelength.

        It must be overwritten by each specific attenuation model,
        and it should returnan (Nmodels, Nwave) array.

        In this example it's assumed that the shape of the curve is
        unmodified and that the normalization is multiplicative. In
        practice this can be more complex.
        '''

        if len(params.shape) == 1:
            params = params.reshape(1,-1)

        ob = self._check_bounds(params)
        if (np.any(ob)):
            raise ValueError('Given parameters are out of bounds for this model (%s).' % (self.model_name))

        if self.Nparams is not None:
            assert (self.Nparams == params.shape[1]), "Number of parameters must match the number (%d) expected by this model (%s)" % (self.Nparams, self.model_name)
        Nmodels = params.shape[0]

        norm = np.zeros(Nmodels)

        expminustau = norm[:,None] * self.expminustau_normed[None,:]

        if Nmodels == 1:
            expminustau = expminustau.flatten()

        return expminustau
