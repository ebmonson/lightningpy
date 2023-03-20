#!/usr/bin/env python

'''
    dust.py

    Class interfaces for dust attenuation and emission modeling.
    Ported from IDL Lightning.

    TODO:
    - Update extrapolation of Calzetti curves to new method
'''

from pathlib import Path
import numpy as np
from scipy.integrate import trapz
from scipy.interpolate import interp1d
import astropy.constants as const
import astropy.units as u

__all__ = ['modified_calzetti', 'DustModel', 'ModifiedCalzettiAtten']

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

    def check_bounds(self, params):
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

        ob = self.check_bounds(params)
        if (np.any(ob)):
            raise ValueError('Given parameters are out of bounds for this model (%s).' % (self.model_name))

        if self.Nparams is not None:
            assert (self.Nparams == params.shape[1]), "Number of parameters must match the number (%d) expected by this model (%s)" % (self.Nparams, self.model_name)
        Nmodels = params.shape[0]

        expminustau = np.zeros((Nmodels, len(self.age)))

        if Nmodels == 1:
            expminustau = expminustau.flatten()

        return expminustau


class CalzettiAtten(AnalyticAtten):

    type = 'analytic'
    model_name = 'Calzetti'
    Nparams = 1
    param_names = ['calz_tauV_diff']
    param_descr = ['Optical depth of the diffuse ISM']
    param_bounds = np.array([0, np.inf]).reshape(1,2)

    def __init__(self, wave):

        self.wave = wave
        self.Nwave = len(self.wave)


    def evaluate(self, params):
        '''
            Model includes a featureless Calzetti law.
        '''

        klam = np.zeros_like(self.wave) # k_lambda, the opacity as a function of wavelength?
        flam1 = np.zeros_like(self.wave)
        flam2 = np.zeros_like(self.wave)

        # Different in this case since there is only one parameter
        if (len(params.shape) == 1):
            params = params.reshape(-1,1)

        params_shape = params.shape
        Nmodels = params_shape[0]
        assert (self.Nparams == params_shape[1]), "Number of parameters must match the number (%d) expected by this model (%s)" % (self.Nparams, self.model_name)

        ob = self.check_bounds(params)
        if (np.any(ob)):
            raise ValueError('Given parameters are out of bounds for this model (%s).' % (self.model_name))

        tauV_diff = params[:,0]

        RV = 4.05

        # Calzetti curve extrapolated from 0.01 to 5.0 microns
        # originally derived from 0.0912 to 2.2 microns
        w1 = (self.wave >= 0.6300) & (self.wave <= 2.2000)
        w2 = (self.wave >= 0.0912) & (self.wave <= 0.6300)
        w3 = (self.wave < 0.0912) & (self.wave > 2.2000)

        kk = 1./self.wave

        klam[w1] = 2.659 * (-1.857 + 1.040 * kk[w1]) + RV
        klam[w2] = 2.659 * np.polynomial.Polynomial([-2.156, 1.509, -0.198, 0.011])(kk[w2]) + RV
        klam[w3] = 0.0

        flam1 = klam[None,:] / 4.05

        expminustau = np.exp(-1 * tauV_diff[:,None] * flam1)

        if (Nmodels == 1):
            expminustau = expminustau.flatten()


        return expminustau


class ModifiedCalzettiAtten(AnalyticAtten):

    type = 'analytic'
    model_name = 'Modified-Calzetti'
    Nparams = 3
    param_names = ['mcalz_tauV_diff', 'mcalz_delta', 'mcalz_tauV_BC']
    param_descr = ['Optical depth of the diffuse ISM',
                   'Deviation from the Calzetti+2000 UV power law slope',
                   'Optical depth of the birth cloud in star forming regions']
    param_bounds = np.array([[0, np.inf],
                             [-np.inf, np.inf],
                             [0, np.inf]])

    def __init__(self, wave):

        self.wave = wave
        self.Nwave = len(self.wave)


    def evaluate(self, params):
        '''
            Model includes a featureless Calzetti law, with the
            addition of a UV bump at 2175 A and optionally extra birth cloud
            extinction. The same attenuation model used in most cases by
            Lightning; I ported it from the Lightning IDL source.

            As of right now the birth cloud component should be ignored, it isn't really
            implemented properly at the moment -- it'll be applied to all ages if you set
            tauV_BC > 0.

            If I were willing to be a little more clever, I would define this more obviously
            as an extension of the CalzettiAtten class.
        '''

        klam = np.zeros_like(self.wave)
        flam1 = np.zeros_like(self.wave)
        flam2 = np.zeros_like(self.wave)

        if (len(params.shape) == 1):
            params = params.reshape(1,-1)

        params_shape = params.shape
        Nmodels = params_shape[0]
        assert (self.Nparams == params_shape[1]), "Number of parameters must match the number (%d) expected by this model (%s)" % (self.Nparams, self.model_name)

        ob = self.check_bounds(params)
        if (np.any(ob)):
            raise ValueError('Given parameters are out of bounds for this model (%s).' % (self.model_name))

        tauV_diff = params[:,0]
        delta = params[:,1]
        tauV_BC = params[:,2]

        RV = 4.05
        FWHM_bump = 0.0350 # 350 A
        lam0_bump = 0.2175 # 2175 A
        # Modifications as in Noll et al. 2009
        Dlam = (0.85 - 1.9 * delta[:,None]) / (((self.wave[None,:]**2 - lam0_bump**2) / (self.wave[None,:] * FWHM_bump))**2 + 1.)

        # Calzetti curve extrapolated from 0.01 to 5.0 microns
        # originally derived from 0.0912 to 2.2 microns
        w1 = (self.wave >= 0.6300) & (self.wave <= 2.2000)
        w2 = (self.wave >= 0.0912) & (self.wave <= 0.6300)
        w3 = (self.wave < 0.0912) & (self.wave > 2.2000)

        kk = 1./self.wave

        klam[w1] = 2.659 * (-1.857 + 1.040 * kk[w1]) + RV
        klam[w2] = 2.659 * np.polynomial.Polynomial([-2.156, 1.509, -0.198, 0.011])(kk[w2]) + RV
        klam[w3] = 0.0

        flam1 = (klam[None,:] + Dlam) / 4.05 * (self.wave[None,:] / 0.55)**delta[:,None]

        # if (tauV_BC != 0):
        #     flam2  = 0.55 / wave # For birth cloud attenuation, somehow
        #

        flam2  = 0.55 / self.wave

        expminustau = np.exp(-1*tauV_diff[:,None] * flam1 + -1*tauV_BC[:,None] * flam2[None,:])
        if (Nmodels == 1):
            expminustau = expminustau.flatten()


        return expminustau


def modified_calzetti(wave, tauV_diff, delta, tauV_BC):
    '''
        Return e^-tau extinction curve as a function of
        *wave*.

        Model includes a featureless Calzetti law, with the
        addition of a UV bump at 2175 A and optionally extra birth cloud
        extinction. The same attenuation model used in most cases by
        Lightning; I ported it from the Lightning IDL source.
    '''

    klam = np.zeros_like(wave) # k_lambda, the opacity as a function of wavelength?
    flam1 = np.zeros_like(wave) # F_lambda flux density
    flam2 = np.zeros_like(wave)

    wave = np.array(wave)
    tauV_diff = np.array(tauV_diff)
    if (tauV_diff.size == 1): tauV_diff = tauV_diff.reshape(1,)
    delta = np.array(delta)
    if (delta.size == 1): delta = delta.reshape(1,)
    tauV_BC = np.array(tauV_BC)
    if (tauV_BC.size == 1): tauV_BC = tauV_BC.reshape(1,)
    Nmodels = tauV_diff.size
    assert (delta.size == Nmodels), 'All parameter arrays must have the same length for vectorization'
    assert (tauV_BC.size == Nmodels), 'All parameter arrays must have the same length for vectorization'

    RV = 4.05
    FWHM_bump = 0.0350 # 350 A
    lam0_bump = 0.2175   # 2175 A
    # Modifications as in Noll et al. 2009
    Dlam = (0.85 - 1.9 * delta[:,None]) / (((wave[None,:]**2 - lam0_bump**2) / (wave[None,:] * FWHM_bump))**2 + 1.)

    # Calzetti curve extrapolated from 0.01 to 5.0 microns
    # originally derived from 0.0912 to 2.2 microns
    w1 = (wave >= 0.6300) & (wave <= 2.2000)
    w2 = (wave >= 0.0912) & (wave <= 0.6300)
    w3 = (wave < 0.0912) & (wave > 2.2000)

    # Calzetti curve is often defined
    # in terms of inverse wavelength.
    kk = 1./wave

    klam[w1] = 2.659 * (-1.857 + 1.040 * kk[w1]) + RV
    klam[w2] = 2.659 * np.polynomial.Polynomial([-2.156, 1.509, -0.198, 0.011])(kk[w2]) + RV
    klam[w3] = 0.0

    flam1 = (klam[None,:] + Dlam) / 4.05 * (wave[None,:] / 0.55)**delta[:,None]

    # if (tauV_BC != 0):
    #     flam2  = 0.55 / wave # For birth cloud attenuation, somehow
    #

    flam2  = 0.55 / wave

    exp_tau = np.exp(-1*tauV_diff[:,None] * flam1 + -1*tauV_BC[:,None] * flam2[None,:])
    if (Nmodels == 1):
        exp_tau = exp_tau.flatten()


    return exp_tau


#################################
# Dust Emission
#################################

class DustModel:
    '''
        An implementation of the Draine & Li (2007) dust emission models.

        A portion gamma of the dust is exposed to a radiation field such
        that the mass of dust dM exposed to a range of intensities [U, U + dU] is
        dM propto U ^ (-alpha) dU, where U is in [Umin, Umax]. The remaining portion
        (1 - gamma) is exposed to a radiation field with intensity Umin.
    '''

    Nparams = 5
    model_name = 'DL07-Dust'
    param_names = ['dl07_dust_alpha', 'dl07_dust_U_min', 'dl07_dust_U_max', 'dl07_dust_gamma', 'dl07_dust_q_PAH']
    param_descr = ['Radiation field intensity distribution power law index',
                   'Radiation field minimum intensity',
                   'Radiation field maximum intensity',
                   'Fraction of dust mass exposed to radiation field intensity distribution',
                   'Fraction of dust mass composed of PAH grains']
    param_bounds = np.array([[-10.0, 4.0],
                             [0.1, 25.0],
                             [1.0e3, 3.0e5],
                             [0.0, 1.0],
                             [0.0047, 0.0458]])

    def __init__(self, filter_labels, redshift,
                 path_to_filters=None,
                 path_to_models=None):

        self.redshift = redshift
        self.filter_labels = filter_labels

        if (path_to_models is None):
            self.path_to_models = str(Path(__file__).parent.resolve()) + '/models/dust/DL07/'
        else:
            self.path_to_models = path_to_models
            if(self.path_to_models[-1] != '/'): self.path_to_models = self.path_to_models + '/'


        if (path_to_filters is None):
            self.path_to_filters = str(Path(__file__).parent.resolve()) + '/filters/'
        else:
            self.path_to_filters = path_to_filters
            if(self.path_to_filters[-1] != '/'): self.path_to_filters = self.path_to_filters + '/'


        self._U_grid = ['0.10','0.15','0.20','0.30','0.40','0.50','0.70','0.80',
                      '1.00','1.20','1.50','2.50','3.00','4.00', '5.00','7.00',
                      '8.00','12.0','15.0','20.0','25.0','1e2','3e2','1e3',
                      '3e3','1e4','3e4','1e5','3e5']
        self._mod_grid = ['MW3.1_00', 'MW3.1_10', 'MW3.1_20', 'MW3.1_30',
                         'MW3.1_40', 'MW3.1_50', 'MW3.1_60']
        self._qPAH_grid = [0.0047, 0.012, 0.0177, 0.0250, 0.0319, 0.0390, 0.0470]

        n_U = len(self._U_grid)
        n_mod = len(self._mod_grid)

        self.Lnu_rest = np.zeros((n_U, n_mod, 1001), dtype='double')
        self.Lbol = np.zeros((n_U, n_mod), dtype='double')

        c_um = const.c.to(u.micron / u.s).value

        for i,U in enumerate(self._U_grid):
            for j,mod in enumerate(self._mod_grid):

                fname = self.path_to_models + 'U%s/U%s_%s_%s.txt' % (U,U,U,mod)

                model_arr = np.loadtxt(fname, usecols=(0,1), skiprows=61)

                # Model files are in order of decreasing wavelenth/increasing nu
                nu = c_um / model_arr[:,0]

                self.Lnu_rest[i,j,:] = model_arr[:,1][::-1] / nu[::-1]
                nu = nu[::-1]

                self.Lbol[i,j] = np.abs(trapz(self.Lnu_rest[i,j,:], nu))

        self.nu_grid_rest = nu
        self.nu_grid_obs = (1 + self.redshift) * self.nu_grid_rest
        self.wave_grid_rest = model_arr[:,0][::-1]
        self.wave_grid_obs = (1 + self.redshift) * self.wave_grid_rest
        self.Lnu_obs = (1 + self.redshift) * self.Lnu_rest

        # Get filters
        self.filters = dict()
        for name in self.filter_labels:
            self.filters[name] = np.zeros(len(self.wave_grid_rest), dtype='float')

        self._get_filters()
        self.Nfilters = len(self.filters)

        self.wave_obs = np.zeros(len(self.filter_labels), dtype='float')
        self._get_wave_obs()
        self.nu_obs = c_um / self.wave_obs

        Lnu_flat = self.Lnu_obs.reshape(-1, len(self.wave_grid_rest))

        mean_Lnu = np.zeros((n_U * n_mod, self.Nfilters))

        for i, filter_label in enumerate(self.filters):

            # Recall that the filters are normalized to 1 when integrated against wave_grid_obs
            # so we can integrate with that to get the mean Lnu in each band.

            mean_Lnu[:,i] = trapz(self.filters[filter_label][None,:] * Lnu_flat, self.wave_grid_obs, axis=1)


        self.mean_Lnu = mean_Lnu.reshape(n_U, n_mod, self.Nfilters)


    def _get_filters(self):
        '''
            Load the filters.
        '''

        from .get_filters import get_filters

        self.filters = get_filters(self.filter_labels, self.wave_grid_obs, self.path_to_filters)


    def _get_wave_obs(self):
        '''
            Compute the mean wavelength of the normalized filters.
        '''

        for i, label in enumerate(self.filter_labels):
            lam = trapz(self.wave_grid_obs * self.filters[label], self.wave_grid_obs)
            self.wave_obs[i] = lam

    def check_bounds(self, params):
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

    def get_model_lnu(self, params):

        param_shape = params.shape # expecting ndarray(Nmodels, Nparams)
        if (len(param_shape) == 1):
            params = params.reshape(1, params.size)
            param_shape = params.shape


        Nmodels = param_shape[0]
        Nparams_in = param_shape[1]

        assert (self.Nparams == Nparams_in), 'The dust model has 5 parameters'

        ob = self.check_bounds(param)
        if(np.any(ob)):
            raise ValueError('Given parameters are out of bounds for this model (%s).' % (self.model_name))

        #Lbol = np.zeros(Nmodels, len(self._U_grid))
        #Lnu = np.zeros(Nmodels, self.Nfilters, len(self._U_grid))

        finterp_Lbol_qPAH = interp1d(self._qPAH_grid, self.Lbol, axis=1)
        Lbol = finterp_Lbol_qPAH(params[:,4]) # ndarray(len(self._U_grid), Nmodels)
        finterp_Lnu_qPAH = interp1d(self._qPAH_grid, self.mean_Lnu, axis=1)
        Lnu = finterp_Lnu_qPAH(params[:,4]) # ndarray(len(self._U_grid), Nmodels, Nfilters)

        # Power law component U^(-alpha)
        U_float = np.array(self._U_grid, dtype='float')
        plaw = U_float[:,None] ** (-1 * params[:,0]) # ndarray(len(self._U_grid), Nmodels)
        # For each model, zero out power law outside of [U_min, U_max]
        mask = (U_float[:,None] < params[:,1]) | (U_float[:,None] > params[:,2])
        plaw[mask] = 0
        plaw = plaw / trapz(plaw, U_float, axis=0) # Normalize

        Lbol_pow = params[:,3] * trapz(plaw * Lbol, U_float, axis=0)
        Lnu_pow = params[:,3][:,None] * trapz(plaw[:,:,None] * Lnu, U_float, axis=0)

        # Delta function component U_max = U_min
        Lbol_delta = np.zeros(Nmodels)
        Lnu_delta = np.zeros((Nmodels, self.Nfilters))
        # Should be able to do this without the loop but I haven't figured out the correct index trickery
        for i in np.arange(Nmodels):
            finterp_Lbol_U = interp1d(U_float, Lbol[:,i], axis=0)
            #Lbol_delta = (1 - params[:,3]) * finterp_Lbol_U(params[:,1]) # ndarray(Nmodels)
            Lbol_delta[i] = finterp_Lbol_U(params[i,1]) # ndarray(Nmodels)
            finterp_Lnu_U = interp1d(U_float, Lnu[:,i,:], axis=0)
            # Lnu_delta = (1 - params[:,3])[:,None] * finterp_Lnu_U(params[:,1])
            Lnu_delta[i,:] = finterp_Lnu_U(params[i,1])

        Lnu_obs = Lnu_pow + Lnu_delta
        Lbol = Lbol_pow + Lbol_delta

        return Lnu_obs, Lbol


    def get_model_lnu_hires(self, params):

        param_shape = params.shape # expecting ndarray(Nmodels, Nparams)
        if (len(param_shape) == 1):
            params = params.reshape(1, params.size)
            param_shape = params.shape


        Nmodels = param_shape[0]
        Nparams_in = param_shape[1]

        assert (self.Nparams == Nparams_in), 'The dust model has 5 parameters'

        ob = self.check_bounds(params)
        if(np.any(ob)):
            raise ValueError('Given parameters are out of bounds for this model (%s).' % (self.model_name))

        #Lbol = np.zeros(Nmodels, len(self._U_grid))
        #Lnu = np.zeros(Nmodels, self.Nfilters, len(self._U_grid))

        finterp_Lbol_qPAH = interp1d(self._qPAH_grid, self.Lbol, axis=1)
        Lbol = finterp_Lbol_qPAH(params[:,4]) # ndarray(len(self._U_grid), Nmodels)
        finterp_Lnu_qPAH = interp1d(self._qPAH_grid, self.Lnu_obs, axis=1)
        Lnu = finterp_Lnu_qPAH(params[:,4]) # ndarray(len(self._U_grid), Nmodels, len(self.wave_grid_rest))

        # Power law component U^(-alpha)
        U_float = np.array(self._U_grid, dtype='float')
        plaw = U_float[:,None] ** (-1 * params[:,0]) # ndarray(len(self._U_grid), Nmodels)
        # For each model, zero out power law outside of [U_min, U_max]
        mask = (U_float[:,None] < params[:,1]) | (U_float[:,None] > params[:,2])
        plaw[mask] = 0
        plaw = plaw / trapz(plaw, U_float, axis=0) # Normalize

        Lbol_pow = params[:,3] * trapz(plaw * Lbol, U_float, axis=0)
        Lnu_pow = params[:,3][:,None] * trapz(plaw[:,:,None] * Lnu, U_float, axis=0)

        # Delta function component U_max = U_min
        Lbol_delta = np.zeros(Nmodels)
        Lnu_delta = np.zeros((Nmodels, len(self.wave_grid_rest)))
        for i in np.arange(Nmodels):
            finterp_Lbol_U = interp1d(U_float, Lbol[:,i], axis=0)
            #Lbol_delta = (1 - params[:,3]) * finterp_Lbol_U(params[:,1]) # ndarray(Nmodels)
            Lbol_delta[i] = finterp_Lbol_U(params[i,1]) # ndarray(Nmodels)
            finterp_Lnu_U = interp1d(U_float, Lnu[:,i,:], axis=0)
            # Lnu_delta = (1 - params[:,3])[:,None] * finterp_Lnu_U(params[:,1])
            Lnu_delta[i,:] = finterp_Lnu_U(params[i,1])


        Lnu_obs = Lnu_pow + Lnu_delta
        Lbol = Lbol_pow + Lbol_delta

        return Lnu_obs, Lbol
