import numpy as np

from ..attenuation import TabulatedAtten

class SMC(TabulatedAtten):

    type = 'tabulated'
    model_name = 'smc'
    Nparams = 1
    param_names = ['smc_tauV_diff']
    param_descr = ['Optical depth of the diffuse ISM']
    param_names_fncy = [r'$\tau_{V,\rm diff}$']
    param_bounds = np.array([0, 1e5]).reshape(1,2)
    path = 'dust/att/smc.txt'

    def evaluate(self, params):

        params = np.array(params)

        if len(params.shape) <= 1:
            params = params.reshape(-1, 1)

        ob = self._check_bounds(params)
        if (np.any(ob)):
            raise ValueError('Given parameters are out of bounds for this model (%s).' % (self.model_name))

        if self.Nparams is not None:
            assert (self.Nparams == params.shape[1]), "Number of parameters must match the number (%d) expected by this model (%s)" % (self.Nparams, self.model_name)
        Nmodels = params.shape[0]

        expminustau = self.expminustau_normed[None,:] ** params

        # Make the model opaque to LyC emission
        expminustau[:, self.wave < 0.0912] = 0

        if Nmodels == 1:
            expminustau = expminustau.flatten()

        return expminustau
