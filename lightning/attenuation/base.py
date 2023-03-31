import numpy as np

#################################
# Dust Attenuation
#################################
class AnalyticAtten:
    '''
        Base class for analytic (i.e., not empirical, tabulated, etc.)
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

        expminustau = np.zeros((Nmodels, len(self.age)))

        if Nmodels == 1:
            expminustau = expminustau.flatten()

        return expminustau
