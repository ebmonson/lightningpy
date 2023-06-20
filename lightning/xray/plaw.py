import numpy as np

from scipy.integrate import trapz
# from scipy.interpolate import interp1d

import astropy.constants as const
import astropy.units as u

from .base import XrayEmissionModel

def plaw(E, norm, gamma):

    return norm[:,None] * E[None,:] ** (1 - gamma[:,None])

def plaw_expcut(E, norm, gamma, E_cut):

    return norm[:,None] * np.exp(-1 * E[None,:] / E_cut[:,None]) * E[None,:] ** (1 - gamma[:,None])

class XrayPlaw(XrayEmissionModel):

    Nparams = 2
    model_name = 'Xray-Plaw'
    model_type = 'analytic'
    gridded = False
    param_names = ['Norm', 'PhoIndex']
    param_descr = ['Normalization at 1 keV (rest-frame)',
                   'Photon index']
    param_names_fncy = [r'$\rm Norm_{plaw}$', r'$\Gamma_{\rm plaw}$']
    param_bounds = np.array([[0, np.inf],
                             [-2.0, 4.0]])

    def _construct_model(self, wave_grid=(1e-6, 1e-1, 200), **kwargs):
        '''
            Build the model from the appropriate files, a function, whatever.
            Here, this just defines the basics.
        '''

        if(isinstance(wave_grid, tuple)):
            self.wave_grid_rest = np.logspace(np.log10(wave_grid[0]), np.log10(wave_grid[1]), wave_grid[2])
        elif(wave_grid is None):
            self.wave_grid_rest = np.logspace(-6, -1, 200)
        else:
            self.wave_grid_rest = wave_grid

        #self._func = lambda E, gamma, A: A * E ** (1 - gamma)

    def get_model_lnu_hires(self, params, exptau=None):

        param_shape = params.shape # expecting ndarray(Nmodels, Nparams)
        if (len(param_shape) == 1):
            params = params.reshape(1, params.size)
            param_shape = params.shape

        Nmodels = param_shape[0]
        Nparams_in = param_shape[1]

        assert (self.Nparams == Nparams_in), 'The %s model has %d parameters' % (self.model_name, self.Nparams)

        ob = self._check_bounds(params)
        if(np.any(ob)):
            raise ValueError('Given parameters are out of bounds for this model (%s).' % (self.model_name))

        norm = params[:,0]
        gamma = params[:,1]

        lnu = (1 + self.redshift) * plaw(self.energ_grid_rest, norm, gamma)

        if (Nmodels == 1):
            lnu = lnu.flatten()

        return lnu

    def get_model_countrate_hires(self, params, exptau=None):

        param_shape = params.shape # expecting ndarray(Nmodels, Nparams)
        if (len(param_shape) == 1):
            params = params.reshape(1, params.size)
            param_shape = params.shape

        Nmodels = param_shape[0]

        lnu_obs = self.get_model_lnu_hires(params)

        # Trying to avoid underflow
        countrate = np.log10(1 / (4 * np.pi)) - 2 * np.log10(self._DL_cm) + \
                    np.log10(lnu_obs) + np.log10(self.specresp) - np.log10(self.phot_energ)

        # print(np.count_nonzero(self.phot_energ))
        # print(np.count_nonzero(lnu_rest))
        # print(np.count_nonzero(countrate))
        # print(np.count_nonzero(self.specresp))
        # print(np.amin(countrate), np.amax(countrate))
        # print(self._DL_cm)

        countrate = 10 ** countrate

        # print(np.count_nonzero(countrate))

        if (Nmodels == 1):
            countrate = countrate.flatten()

        return countrate

    def get_model_lnu(self, params, exptau=None):

        param_shape = params.shape # expecting ndarray(Nmodels, Nparams)
        if (len(param_shape) == 1):
            params = params.reshape(1, params.size)
            param_shape = params.shape

        Nmodels = param_shape[0]
        Nparams_in = param_shape[1]

        assert (self.Nparams == Nparams_in), 'The %s model has %d parameters' % (self.model_name, self.Nparams)

        ob = self._check_bounds(params)
        if(np.any(ob)):
            raise ValueError('Given parameters are out of bounds for this model (%s).' % (self.model_name))

        lnu_hires = self.get_model_lnu_hires(params, exptau=None)

        if (Nmodels == 1):
            lnu_hires = lnu_hires.reshape(1,-1)

        lmod = np.zeros((Nmodels, self.Nfilters))

        for i, filter_label in enumerate(self.filters):
            # Recall that the filters are normalized to 1 when integrated against wave_grid_obs
            # so we can integrate with that to get the mean Lnu in each band.
            lmod[:,i] = trapz(self.filters[filter_label][None,:] * lnu_hires, self.wave_grid_obs, axis=1)

        return lmod

    def get_model_countrate(self, params, exptau=None):

        param_shape = params.shape # expecting ndarray(Nmodels, Nparams)
        if (len(param_shape) == 1):
            params = params.reshape(1, params.size)
            param_shape = params.shape

        Nmodels = param_shape[0]
        Nparams_in = param_shape[1]

        assert (self.Nparams == Nparams_in), 'The %s model has %d parameters' % (self.model_name, self.Nparams)

        ob = self._check_bounds(params)
        if(np.any(ob)):
            raise ValueError('Given parameters are out of bounds for this model (%s).' % (self.model_name))

        countrate_hires = self.get_model_countrate_hires(params, exptau=None)

        if (Nmodels == 1):
            lnu_hires = lnu_hires.reshape(1,-1)

        countrate = np.zeros((Nmodels, self.Nfilters))

        for i, filter_label in enumerate(self.filters):
            # Recall that the filters are normalized to 1 when integrated against wave_grid_obs
            # so we can integrate with that to get the mean countrate in each band.
            countrate[:,i] = trapz(self.filters[filter_label][None,:] * countrate_hires, self.wave_grid_obs, axis=1)

        return countrate

    def get_model_counts(self, params, exptau=None):

        param_shape = params.shape # expecting ndarray(Nmodels, Nparams)
        if (len(param_shape) == 1):
            params = params.reshape(1, params.size)
            param_shape = params.shape

        Nmodels = param_shape[0]
        Nparams_in = param_shape[1]

        assert (self.Nparams == Nparams_in), 'The %s model has %d parameters' % (self.model_name, self.Nparams)

        ob = self._check_bounds(params)
        if(np.any(ob)):
            raise ValueError('Given parameters are out of bounds for this model (%s).' % (self.model_name))

        # This is the mean count-rate density, counts s-1 Hz-1
        countrate = self.get_model_countrate(params, exptau=None)

        # The counts are the width of the bandpass times the mean countrate density
        # times the exposure time. The definition of the "width" of an arbitrary bandpass
        # is not terrible important, since the countrate spectrum is not defined outside of
        # where we define the ARF.
        counts = np.zeros((Nmodels, self.Nfilters))
        for i, filter_label in enumerate(self.filters):
            # This isn't the best way to do it though
            nu_max = np.amax(self.nu_grid_obs[np.nonzero(self.filters[filter_label])])
            nu_min = np.amin(self.nu_grid_obs[np.nonzero(self.filters[filter_label])])

            print(nu_min, nu_max)

            counts[:,i] = self.exposure * (nu_max - nu_min) * countrate[:,i]

        return counts

class XrayPlawExpcut(XrayEmissionModel):

    Nparams = 3
    model_name = 'Xray-Plaw-Expcut'
    model_type = 'analytic'
    gridded = False
    param_names = ['Norm', 'PhoIndex', 'E_cut']
    param_descr = ['Normalization at 1 keV (rest-frame)',
                   'Photon index',
                   'High energy cutoff location']
    param_names_fncy = [r'$\rm Norm_{plaw}$', r'$\Gamma_{\rm plaw}$', r'$E_{\rm cut}$']
    param_bounds = np.array([[0, np.inf],
                             [-2.0, 4.0],
                             [0.0, np.inf]])

    def _construct_model(self, wave_grid=(1e-6, 1e-1, 200), **kwargs):
        '''
            Build the model from the appropriate files, a function, whatever.
            Here, this just defines the basics.
        '''

        if(isinstance(wave_grid, tuple)):
            self.wave_grid_rest = np.logspace(np.log10(wave_grid[0]), np.log10(wave_grid[1]), wave_grid[2])
        elif(wave_grid is None):
            self.wave_grid_rest = np.logspace(-6, -1, 200)
        else:
            self.wave_grid_rest = wave_grid

        #self._func = lambda E, gamma, A: A * E ** (1 - gamma)

    def get_model_lnu_hires(self, params, exptau=None):

        param_shape = params.shape # expecting ndarray(Nmodels, Nparams)
        if (len(param_shape) == 1):
            params = params.reshape(1, params.size)
            param_shape = params.shape

        Nmodels = param_shape[0]
        Nparams_in = param_shape[1]

        assert (self.Nparams == Nparams_in), 'The %s model has %d parameters' % (self.model_name, self.Nparams)

        ob = self._check_bounds(params)
        if(np.any(ob)):
            raise ValueError('Given parameters are out of bounds for this model (%s).' % (self.model_name))

        norm = params[:,0]
        gamma = params[:,1]
        E_cut = params[:,2]

        lnu = (1 + self.redshift) * plaw_expcut(self.energ_grid_rest, norm, gamma, E_cut)

        if (Nmodels == 1):
            lnu = lnu.flatten()

        return lnu

    def get_model_countrate_hires(self, params, exptau=None):

        param_shape = params.shape # expecting ndarray(Nmodels, Nparams)
        if (len(param_shape) == 1):
            params = params.reshape(1, params.size)
            param_shape = params.shape

        Nmodels = param_shape[0]

        lnu_obs = self.get_model_lnu_hires(params)

        countrate = np.log10(1 / (4 * np.pi * self._DL_cm ** 2)) + \
                    np.log10(lnu_obs) + np.log10(self.specresp) - np.log10(self.phot_energ)

        countrate = 10 ** countrate

        if (Nmodels == 1):
            countrate = countrate.flatten()

        return countrate

    def get_model_lnu(self, params, exptau=None):

        param_shape = params.shape # expecting ndarray(Nmodels, Nparams)
        if (len(param_shape) == 1):
            params = params.reshape(1, params.size)
            param_shape = params.shape

        Nmodels = param_shape[0]
        Nparams_in = param_shape[1]

        assert (self.Nparams == Nparams_in), 'The %s model has %d parameters' % (self.model_name, self.Nparams)

        ob = self._check_bounds(params)
        if(np.any(ob)):
            raise ValueError('Given parameters are out of bounds for this model (%s).' % (self.model_name))

        lnu_hires = self.get_model_lnu_hires(params, exptau=None)

        if (Nmodels == 1):
            lnu_hires = lnu_hires.reshape(1,-1)

        lmod = np.zeros((Nmodels, self.Nfilters))

        for i, filter_label in enumerate(self.filters):
            # Recall that the filters are normalized to 1 when integrated against wave_grid_obs
            # so we can integrate with that to get the mean Lnu in each band.
            lmod[:,i] = trapz(self.filters[filter_label][None,:] * lnu_hires, self.wave_grid_obs, axis=1)

        return lmod

    def get_model_countrate(self, params, exptau=None):

        param_shape = params.shape # expecting ndarray(Nmodels, Nparams)
        if (len(param_shape) == 1):
            params = params.reshape(1, params.size)
            param_shape = params.shape

        Nmodels = param_shape[0]
        Nparams_in = param_shape[1]

        assert (self.Nparams == Nparams_in), 'The %s model has %d parameters' % (self.model_name, self.Nparams)

        ob = self._check_bounds(params)
        if(np.any(ob)):
            raise ValueError('Given parameters are out of bounds for this model (%s).' % (self.model_name))

        countrate_hires = self.get_model_countrate_hires(params, exptau=None)

        if (Nmodels == 1):
            lnu_hires = lnu_hires.reshape(1,-1)

        countrate = np.zeros((Nmodels, self.Nfilters))

        for i, filter_label in enumerate(self.filters):
            # Recall that the filters are normalized to 1 when integrated against wave_grid_obs
            # so we can integrate with that to get the mean countrate in each band.
            countrate[:,i] = trapz(self.filters[filter_label][None,:] * countrate_hires, self.wave_grid_obs, axis=1)

        return countrate

    def get_model_counts(self, params, exptau=None):

        param_shape = params.shape # expecting ndarray(Nmodels, Nparams)
        if (len(param_shape) == 1):
            params = params.reshape(1, params.size)
            param_shape = params.shape

        Nmodels = param_shape[0]
        Nparams_in = param_shape[1]

        assert (self.Nparams == Nparams_in), 'The %s model has %d parameters' % (self.model_name, self.Nparams)

        ob = self._check_bounds(params)
        if(np.any(ob)):
            raise ValueError('Given parameters are out of bounds for this model (%s).' % (self.model_name))

        # This is the mean count-rate density, counts s-1 Hz-1
        countrate = self.get_model_countrate(params, exptau=None)

        # The counts are the width of the bandpass times the mean countrate density
        # times the exposure time. The definition of the "width" of an arbitrary bandpass
        # is not terrible important, since the countrate spectrum is not defined outside of
        # where we define the ARF.
        counts = np.zeros((Nmodels, self.Nfilters))
        for i, filter_label in enumerate(self.filters):
            nu_max = np.amax(self.nu_grid_obs[np.nonzero(self.filters[filter_label])])
            nu_min = np.amin(self.nu_grid_obs[np.nonzero(self.filters[filter_label])])

            counts[:,i] = self.exposure * (nu_max - nu_min) * countrate[:,i]

        return counts
