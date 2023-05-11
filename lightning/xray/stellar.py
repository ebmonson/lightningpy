import numpy as np

from scipy.integrate import trapz

from .plaw import plaw_expcut, XrayPlawExpcut

def gilbertson22_L2_10(stellar_model, sfh, sfh_params):
    '''Stellar-age parameterization of the X-ray luminosity.

    From Gilbertson et al. (2022).
    '''

    age = np.array(sfh.age)
    Nmodels = sfh_params.shape[0]

    if (sfh.type == 'piecewise'):

        tau_age = np.log10(0.5 * (age[1:] + age[:-1]))

    else:

        tau_age = np.log10(age)

    # The width of the bandpass in Hz.
    # We might actually lose this; there's not really
    # any point to including this constant only to
    # take it out later.
    #dnu_2_10 = 1.934e18
    Lsun = 3.828e33 # erg s-1

    # log(2-10 keV Lx / Mstar) as a function of time
    loggamma_LMXB = -1.21 * (tau_age - 9.32) ** 2 + 29.09
    loggamma_HMXB = -0.24 * (tau_age - 5.23) ** 2 + 32.54

    # Converted to L in Lsun as a function of time
    L_LMXB_tau = 10**(loggamma_LMXB + np.log10(stellar_model.mstar) - np.log10(Lsun))# - np.log10(dnu_2_10)
    L_HMXB_tau = 10**(loggamma_HMXB + np.log10(stellar_model.mstar) - np.log10(Lsun))# - np.log10(dnu_2_10)

    if (sfh.type == 'piecewise'):

        L_LMXB = sfh.sum(sfh_params, L_LMXB_tau)
        L_HMXB = sfh.sum(sfh_params, L_HMXB_tau)

    else:

        L_LMXB = sfh.integrate(sfh_params, L_LMXB_tau)
        L_HMXB = sfh.integrate(sfh_params, L_HMXB_tau)

    if (Nmodels == 1):
        # promote to 1-length arrays from scalars
        L_LMXB = np.array(L_LMXB).reshape(1,)
        L_HMXB = np.array(L_HMXB).reshape(1,)

    return L_LMXB, L_HMXB


class StellarPlaw(XrayPlawExpcut):
    '''Simple model for stellar X-ray emission.

    Inclues a stellar-age parameterization of the luminosity,
    such that the model normalization is a function of the SFH.
    The high energy cutoff is fixed, but the photon index can
    vary.
    '''
    Nparams = 1
    model_name = 'Stellar-Plaw'
    model_type = 'analytic'
    gridded = False
    param_names = ['PhoIndex']
    param_descr = ['Photon index']
    param_bounds = np.array([[-2.0, 9.0]])

    def get_model_lnu_hires(self, params, stellar_model, sfh, sfh_params, exptau=None):

        # param_shape = params.shape # expecting ndarray(Nmodels, Nparams)
        # if (len(param_shape) == 1):
        #     params = params.reshape(1, params.size)
        #     param_shape = params.shape
        #
        # Nmodels = param_shape[0]
        # Nparams_in = param_shape[1]

        # Different in this case since there is only one parameter
        if (len(params.shape) == 1):
            params = params.reshape(-1,1)

        params_shape = params.shape
        Nmodels = params_shape[0]
        Nparams_in = params_shape[1]
        #assert (self.Nparams == params_shape[1]), "Number of parameters must match the number (%d) expected by this model (%s)" % (self.Nparams, self.model_name)

        assert (self.Nparams == Nparams_in), 'The %s model has %d parameters' % (self.model_name, self.Nparams)

        assert (Nmodels == sfh_params.shape[0]), 'Number of parameter sets must be consistent between power law and SFH.'

        ob = self._check_bounds(params)
        if(np.any(ob)):
            raise ValueError('Given parameters are out of bounds for this model (%s).' % (self.model_name))

        gamma = params[:,0]
        # Fix E_cut at 100 keV
        E_cut = np.full(Nmodels, 100)
        hplanck = 4.136e-18 # keV s

        L_LMXB, L_HMXB = gilbertson22_L2_10(stellar_model, sfh, sfh_params)
        # Recast the rest-frame 2-10 keV luminosity in terms of the normalization.
        # This is more-or-less correct (but not exactly correct) since the
        # exponential cutoff energy is fairly large compared to the bandpass.
        # Note there is a different solution for gamma == 2.
        norm = np.zeros_like(gamma)
        mask = gamma == 2

        norm[~mask] = (2 - gamma[~mask]) * hplanck * (L_LMXB[~mask] + L_HMXB[~mask]) / (10.0**(2-gamma[~mask]) - 2.0**(2-gamma[~mask]))
        if (np.count_nonzero(mask) > 0):
            norm[mask] = hplanck * (L_LMXB[~mask] + L_HMXB[~mask]) / np.log(5.0)

        lnu = (1 + self.redshift) * plaw_expcut(self.energ_grid_rest, norm, gamma, E_cut)

        lnu[:, self.energ_grid_rest < 0.1] = 0

        if (Nmodels == 1):
            lnu = lnu.flatten()

        return lnu

    def get_model_countrate_hires(self, params, stellar_model, sfh, sfh_params, exptau=None):

        if (len(params.shape) == 1):
            params = params.reshape(-1,1)

        params_shape = params.shape
        Nmodels = params_shape[0]
        Nparams_in = params_shape[1]

        assert (self.Nparams == Nparams_in), 'The %s model has %d parameters' % (self.model_name, self.Nparams)

        assert (Nmodels == sfh_params.shape[0]), 'Number of parameter sets must be consistent between power law and SFH.'

        ob = self._check_bounds(params)
        if(np.any(ob)):
            raise ValueError('Given parameters are out of bounds for this model (%s).' % (self.model_name))

        lnu_obs = self.get_model_lnu_hires(params, stellar_model, sfh, sfh_params)

        countrate = np.log10(1 / (4 * np.pi)) - 2 * np.log10(self._DL_cm) + \
                    np.log10(lnu_obs) + np.log10(self.specresp) - np.log10(self.phot_energ)

        countrate = 10 ** countrate

        # print(np.count_nonzero(countrate))

        if (Nmodels == 1):
            countrate = countrate.flatten()

        return countrate

    def get_model_lnu(self, params, stellar_model, sfh, sfh_params, exptau=None):

        if (len(params.shape) == 1):
            params = params.reshape(-1,1)

        params_shape = params.shape
        Nmodels = params_shape[0]
        Nparams_in = params_shape[1]

        assert (self.Nparams == Nparams_in), 'The %s model has %d parameters' % (self.model_name, self.Nparams)

        assert (Nmodels == sfh_params.shape[0]), 'Number of parameter sets must be consistent between power law and SFH.'

        ob = self._check_bounds(params)
        if(np.any(ob)):
            raise ValueError('Given parameters are out of bounds for this model (%s).' % (self.model_name))

        lnu_hires = self.get_model_lnu_hires(params, stellar_model, sfh, sfh_params, exptau=None)

        if (Nmodels == 1):
            lnu_hires = lnu_hires.reshape(1,-1)

        lmod = np.zeros((Nmodels, self.Nfilters))

        for i, filter_label in enumerate(self.filters):
            # Recall that the filters are normalized to 1 when integrated against wave_grid_obs
            # so we can integrate with that to get the mean Lnu in each band.
            lmod[:,i] = trapz(self.filters[filter_label][None,:] * lnu_hires, self.wave_grid_obs, axis=1)

        return lmod

    def get_model_countrate(self, params, stellar_model, sfh, sfh_params, exptau=None):

        if (len(params.shape) == 1):
            params = params.reshape(-1,1)

        params_shape = params.shape
        Nmodels = params_shape[0]
        Nparams_in = params_shape[1]

        assert (self.Nparams == Nparams_in), 'The %s model has %d parameters' % (self.model_name, self.Nparams)

        assert (Nmodels == sfh_params.shape[0]), 'Number of parameter sets must be consistent between power law and SFH.'

        ob = self._check_bounds(params, stellar_model, sfh, sfh_params)
        if(np.any(ob)):
            raise ValueError('Given parameters are out of bounds for this model (%s).' % (self.model_name))

        countrate_hires = self.get_model_countrate_hires(params, stellar_model, sfh, sfh_params, exptau=None)

        if (Nmodels == 1):
            lnu_hires = lnu_hires.reshape(1,-1)

        countrate = np.zeros((Nmodels, self.Nfilters))

        for i, filter_label in enumerate(self.filters):
            # Recall that the filters are normalized to 1 when integrated against wave_grid_obs
            # so we can integrate with that to get the mean countrate in each band.
            countrate[:,i] = trapz(self.filters[filter_label][None,:] * countrate_hires, self.wave_grid_obs, axis=1)

        return countrate

    def get_model_counts(self, params, stellar_model, sfh, sfh_params, exptau=None):

        if (len(params.shape) == 1):
            params = params.reshape(-1,1)

        params_shape = params.shape
        Nmodels = params_shape[0]
        Nparams_in = params_shape[1]

        assert (self.Nparams == Nparams_in), 'The %s model has %d parameters' % (self.model_name, self.Nparams)

        assert (Nmodels == sfh_params.shape[0]), 'Number of parameter sets must be consistent between power law and SFH.'

        ob = self._check_bounds(params)
        if(np.any(ob)):
            raise ValueError('Given parameters are out of bounds for this model (%s).' % (self.model_name))

        # This is the mean count-rate density, counts s-1 Hz-1
        countrate = self.get_model_countrate(params, stellar_model, sfh, sfh_params, exptau=None)

        # The counts are the width of the bandpass times the mean countrate density
        # times the exposure time. The definition of the "width" of an arbitrary bandpass
        # is not terribly important, since the countrate spectrum is not defined outside of
        # where we define the ARF.
        counts = np.zeros((Nmodels, self.Nfilters))
        for i, filter_label in enumerate(self.filters):
            nu_max = np.amax(self.nu_grid_obs[np.nonzero(self.filters[filter_label])])
            nu_min = np.amin(self.nu_grid_obs[np.nonzero(self.filters[filter_label])])

            counts[:,i] = self.exposure * (nu_max - nu_min) * countrate[:,i]

        return counts
