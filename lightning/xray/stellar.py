import numpy as np

from scipy.integrate import trapz

from .plaw import plaw_expcut, XrayPlawExpcut

def gilbertson22_L2_10(stellar_model, stellar_params, sfh, sfh_params):
    '''Stellar-age parameterization of the X-ray luminosity.

    From Gilbertson et al. (2022).
    '''

    age = np.array(sfh.age)
    Nmodels = sfh_params.shape[0]
    Z = stellar_params[:,0]

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
    L_LMXB_tau = 10**(loggamma_LMXB[None,:] + np.log10(stellar_model.get_mstar_coeff(Z)) - np.log10(Lsun))# - np.log10(dnu_2_10)
    L_HMXB_tau = 10**(loggamma_HMXB[None,:] + np.log10(stellar_model.get_mstar_coeff(Z)) - np.log10(Lsun))# - np.log10(dnu_2_10)

    if (sfh.type == 'piecewise'):

        sfh_arr = sfh.evaluate(sfh_params).reshape(Nmodels,-1)

        L_LMXB = np.sum(sfh_arr * L_LMXB_tau, axis=1)
        L_HMXB = np.sum(sfh_arr * L_HMXB_tau, axis=1)

        # L_LMXB = sfh.sum(sfh_params, L_LMXB_tau)
        # L_HMXB = sfh.sum(sfh_params, L_HMXB_tau)

    else:

        sfh_arr = sfh.evaluate(sfh_params).reshape(Nmodels,-1)

        L_LMXB = trapz(sfh_arr * L_LMXB_tau, axis=1)
        L_HMXB = trapz(sfh_arr * L_HMXB_tau, axis=1)

        # L_LMXB = sfh.integrate(sfh_params, L_LMXB_tau)
        # L_HMXB = sfh.integrate(sfh_params, L_HMXB_tau)

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

    The luminosity is determined from the SFH model based on the
    empirical Lx/M - stellar age relationship from Gilbertson+(2022).

    Parameters
    ----------
    filter_labels : list, str
        List of filter labels.
    arf : dict or astropy.table.Table or numpy structured array
        A structure defining the anciliary response function (ARF) of your X-ray observations. The structure must have
        three keys, `'ENERG_LO'`, `'ENERG_HI'`, and `'SPECRESP'`, which given the energy bins and binned spectral response
        respectively. Only used if `xray_mode='counts'``.
    exposure : float or np.ndarray (Nfilters)
        A scalar or array giving the exposure time of the X-ray observations. If an array, it should have the same
        length as ``filter_labels``, with all non-X-ray bands having their exposure time set to 0. Note that you almost
        certainly don't need to give exposure time as an array, since the energy dependence of the effective area
        is explicitly given by the ARF. Only used if ``xray_mode='counts'``.
    redshift : float
        Redshift of the model. If set, ``lum_dist`` is ignored.
    lum_dist : float
        Luminosity distance to the model. If not set, this will
        be calculated from the redshift and cosmology. (Default: None)
    cosmology : astropy.cosmology.FlatLambdaCDM
        The cosmology to assume. Lightning defaults to a flat cosmology with ``h=0.7 and Om0=0.3``.
    path_to_models : str
        Path to lightning models. Not actually used in normal circumstances.
    path_to_filters : str
        Path to lightning filters. Not actually used in normal circumstances.
    wave_grid : tuple (3,), or np.ndarray, (Nwave,), float32, optional
        Either a tuple of (lo, hi, Nwave) specifying a log-spaced rest-frame wavelength grid, or an array
        giving the wavelengths directly. At high redshift this should be constructed carefully to ensure that
        your bands are covered. (Default: (1e-6, 1e-1, 200))

    References
    ----------
    - `Gilbertson et al. (2022) <https://ui.adsabs.harvard.edu/abs/2022ApJ...926...28G/abstract>`_

    '''
    Nparams = 1
    model_name = 'Stellar-Plaw'
    model_type = 'analytic'
    gridded = False
    param_names = ['PhoIndex']
    param_descr = ['Photon index']
    param_names_fncy = [r'$\Gamma_{\rm XRB}$']
    param_bounds = np.array([[-2.0, 9.0]])

    def get_model_lnu_hires(self, params, stellar_model, stellar_params, sfh, sfh_params, exptau=None):
        '''Construct the high-resolution spectrum in Lnu.

        This function takes in the stellar and SFH models in order to use the
        stellar-age parametrization of Lx / M from Gilbertson et al. (2022).

        Parameters
        ----------
        params : np.ndarray(Nmodels, Nparams) or np.ndarray(Nparams)
            An array of model parameters. For purposes of vectorization
            this can be a 2D array, where the first dimension cycles over
            different sets of parameters.
        stellar_model : lightning.stellar model
            The stellar model.
        stellar_params : np.ndarray(Nmodels, Nparams_st) or np.ndarray(Nparams_st)
            The stellar model parameters (i.e. metallicity and possible logU).
        sfh : lightning.sfh.PiecewiseConstSFH or lightning.sfh.FunctionalSFH
            The star formation history model.
        sfh_params : np.ndarray(Nmodels, Nparams_sfh) or np.ndarray(Nparams_sfh)
            The parameters for the SFH.
        exptau : np.ndarray(Nmodels, Nwave) or np.ndarray(Nwave)
            The e(-tau) absorption curve.

        Outputs
        -------
        lnu_abs : np.ndarray(Nmodels, Nwave) or np.ndarray(Nwave)
            The absorbed Lnu spectrum.
        lnu_unabs : np.ndarray(Nmodels, Nwave) or np.ndarray(Nwave)
            The absorbtion-free Lnu spectrum.

        '''

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

        if (exptau is None):
            exptau = np.full((Nmodels,len(self.wave_grid_rest)), 1)
        else:
            exptau_shape = exptau.shape
            assert (exptau_shape[0] == Nmodels), 'Number of parameter sets must be consistent between power law and absorption.'
            assert (exptau_shape[1] == len(self.wave_grid_rest)), 'Number of wavelength elements must be consistent between emission and absorption model.'

        ob = self._check_bounds(params)
        if(np.any(ob)):
            raise ValueError('Given parameters are out of bounds for this model (%s).' % (self.model_name))

        gamma = params[:,0]
        # Fix E_cut at 100 keV
        E_cut = np.full(Nmodels, 100)
        hplanck = 4.136e-18 # keV s

        L_LMXB, L_HMXB = gilbertson22_L2_10(stellar_model, stellar_params, sfh, sfh_params)
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

        lnu_abs = exptau * lnu

        if (Nmodels == 1):
            lnu = lnu.flatten()
            lnu_abs = lnu_abs.flatten()

        return lnu_abs, lnu

    def get_model_countrate_hires(self, params, stellar_model, stellar_params, sfh, sfh_params, exptau=None):
        '''Construct the high-resolution countrate-density spectrum.

        This function takes in the stellar and SFH models in order to use the
        stellar-age parametrization of Lx / M from Gilbertson et al. (2022).

        Parameters
        ----------
        params : np.ndarray(Nmodels, Nparams) or np.ndarray(Nparams)
            An array of model parameters. For purposes of vectorization
            this can be a 2D array, where the first dimension cycles over
            different sets of parameters.
        stellar_model : lightning.stellar model
            The stellar model.
        stellar_params : np.ndarray(Nmodels, Nparams_st) or np.ndarray(Nparams_st)
            The stellar model parameters (i.e. metallicity and possible logU).
        sfh : lightning.sfh.PiecewiseConstSFH or lightning.sfh.FunctionalSFH
            The star formation history model.
        sfh_params : np.ndarray(Nmodels, Nparams_sfh) or np.ndarray(Nparams_sfh)
            The parameters for the SFH.
        exptau : np.ndarray(Nmodels, Nwave) or np.ndarray(Nwave)
            The e(-tau) absorption curve.

        Outputs
        -------
        countrate : np.ndarray(Nmodels, Nwave) or np.ndarray(Nwave)
            Count-rate density in counts s-1 Hz-1.

        Notes
        -----
        - The absorption free count-rate spectrum is not returned by default. It can
          be accessed by setting exptau to ``None``.

        '''

        if (len(params.shape) == 1):
            params = params.reshape(-1,1)

        assert self.specresp is not None, 'ARF must be defined to calculate count rates'

        params_shape = params.shape
        Nmodels = params_shape[0]
        Nparams_in = params_shape[1]

        assert (self.Nparams == Nparams_in), 'The %s model has %d parameters' % (self.model_name, self.Nparams)

        assert (Nmodels == sfh_params.shape[0]), 'Number of parameter sets must be consistent between power law and SFH.'

        ob = self._check_bounds(params)
        if(np.any(ob)):
            raise ValueError('Given parameters are out of bounds for this model (%s).' % (self.model_name))

        # We discard the intrinsic lnu since the 'unabsorbed count-rate' is not really
        # used for anything.
        lnu_obs, _ = self.get_model_lnu_hires(params, stellar_model, stellar_params, sfh, sfh_params, exptau=exptau)

        if (Nmodels == 1):
            lnu_obs = lnu_obs.reshape(1,-1)

        countrate = np.log10(1 / (4 * np.pi)) - 2 * np.log10(self._DL_cm) + \
                    np.log10(lnu_obs) + np.log10(self.specresp[None, :]) - np.log10(self.phot_energ[None,:])

        countrate = 10 ** countrate

        # print(np.count_nonzero(countrate))

        if (Nmodels == 1):
            countrate = countrate.flatten()

        return countrate

    def get_model_lnu(self, params, stellar_model, stellar_params, sfh, sfh_params, exptau=None):
        '''Construct the bandpass-convolved SED in Lnu.

        This function takes in the stellar and SFH models in order to use the
        stellar-age parametrization of Lx / M from Gilbertson et al. (2022).

        Parameters
        ----------
        params : np.ndarray(Nmodels, Nparams) or np.ndarray(Nparams)
            An array of model parameters. For purposes of vectorization
            this can be a 2D array, where the first dimension cycles over
            different sets of parameters.
        stellar_model : lightning.stellar model
            The stellar model.
        stellar_params : np.ndarray(Nmodels, Nparams_st) or np.ndarray(Nparams_st)
            The stellar model parameters (i.e. metallicity and possible logU).
        sfh : lightning.sfh.PiecewiseConstSFH or lightning.sfh.FunctionalSFH
            The star formation history model.
        sfh_params : np.ndarray(Nmodels, Nparams_sfh) or np.ndarray(Nparams_sfh)
            The parameters for the SFH.
        exptau : np.ndarray(Nmodels, Nwave) or np.ndarray(Nwave)
            The e(-tau) absorption curve.

        Outputs
        -------
        lnu_abs : np.ndarray(Nmodels, Nwave) or np.ndarray(Nwave)
            The absorbed Lnu in the bandpasses.
        lnu_unabs : np.ndarray(Nmodels, Nwave) or np.ndarray(Nwave)
            The absorbtion-free Lnu in the bandpasses.

        '''

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

        lnu_obs, lnu_intr = self.get_model_lnu_hires(params, stellar_model, stellar_params, sfh, sfh_params, exptau=exptau)

        if (Nmodels == 1):
            lnu_obs = lnu_obs.reshape(1,-1)
            lnu_intr = lnu_intr.reshape(1,-1)

        lmod = np.zeros((Nmodels, self.Nfilters))
        lmod_intr = np.zeros((Nmodels, self.Nfilters))

        for i, filter_label in enumerate(self.filters):
            # Recall that the filters are normalized to 1 when integrated against wave_grid_obs
            # so we can integrate with that to get the mean Lnu in each band.
            lmod[:,i] = trapz(self.filters[filter_label][None,:] * lnu_obs, self.wave_grid_obs, axis=1)
            lmod_intr[:,i] = trapz(self.filters[filter_label][None,:] * lnu_intr, self.wave_grid_obs, axis=1)

        return lmod, lmod_intr

    def get_model_countrate(self, params, stellar_model, stellar_params, sfh, sfh_params, exptau=None):
        '''Construct the bandpass-convolved SED in count-rate.

        This function takes in the stellar and SFH models in order to use the
        stellar-age parametrization of Lx / M from Gilbertson et al. (2022).

        Parameters
        ----------
        params : np.ndarray(Nmodels, Nparams) or np.ndarray(Nparams)
            An array of model parameters. For purposes of vectorization
            this can be a 2D array, where the first dimension cycles over
            different sets of parameters.
        stellar_model : lightning.stellar model
            The stellar model.
        stellar_params : np.ndarray(Nmodels, Nparams_st) or np.ndarray(Nparams_st)
            The stellar model parameters (i.e. metallicity and possible logU).
        sfh : lightning.sfh.PiecewiseConstSFH or lightning.sfh.FunctionalSFH
            The star formation history model.
        sfh_params : np.ndarray(Nmodels, Nparams_sfh) or np.ndarray(Nparams_sfh)
            The parameters for the SFH.
        exptau : np.ndarray(Nmodels, Nwave) or np.ndarray(Nwave)
            The e(-tau) absorption curve.

        Outputs
        -------
        countrate : np.ndarray(Nmodels, Nfilters) or np.ndarray(Nfilters)
            Count-rate in the bandpasses in counts s-1.

        Notes
        -----
        - The absorption free count-rate is not returned by default. It can
          be accessed by setting exptau to ``None``.

        '''

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

        countrate_hires = self.get_model_countrate_hires(params, stellar_model, stellar_params, sfh, sfh_params, exptau=exptau)

        if (Nmodels == 1):
            countrate_hires = countrate_hires.reshape(1,-1)

        countrate = np.zeros((Nmodels, self.Nfilters))

        for i, filter_label in enumerate(self.filters):
            # Recall that the filters are normalized to 1 when integrated against wave_grid_obs
            # so we can integrate with that to get the mean countrate in each band.
            countrate[:,i] = trapz(self.filters[filter_label][None,:] * countrate_hires, self.wave_grid_obs, axis=1)

        return countrate

    def get_model_counts(self, params, stellar_model, stellar_params, sfh, sfh_params, exptau=None):
        '''Construct the bandpass-convolved SED in counts.

        This function takes in the stellar and SFH models in order to use the
        stellar-age parametrization of Lx / M from Gilbertson et al. (2022).

        Parameters
        ----------
        params : np.ndarray(Nmodels, Nparams) or np.ndarray(Nparams)
            An array of model parameters. For purposes of vectorization
            this can be a 2D array, where the first dimension cycles over
            different sets of parameters.
        stellar_model : lightning.stellar model
            The stellar model.
        stellar_params : np.ndarray(Nmodels, Nparams_st) or np.ndarray(Nparams_st)
            The stellar model parameters (i.e. metallicity and possible logU).
        sfh : lightning.sfh.PiecewiseConstSFH or lightning.sfh.FunctionalSFH
            The star formation history model.
        sfh_params : np.ndarray(Nmodels, Nparams_sfh) or np.ndarray(Nparams_sfh)
            The parameters for the SFH.
        exptau : np.ndarray(Nmodels, Nwave) or np.ndarray(Nwave)
            The e(-tau) absorption curve.

        Outputs
        -------
        counts : np.ndarray(Nmodels, Nfilters) or np.ndarray(Nfilters)
            Counts in the bandpasses.

        Notes
        -----
        - The absorption free counts are not returned by default. They can
          be accessed by setting exptau to ``None``.

        '''

        if (len(params.shape) == 1):
            params = params.reshape(-1,1)

        assert self.exposure is not None, 'Exposure time must be defined to calculate counts'

        params_shape = params.shape
        Nmodels = params_shape[0]
        Nparams_in = params_shape[1]

        assert (self.Nparams == Nparams_in), 'The %s model has %d parameters' % (self.model_name, self.Nparams)

        assert (Nmodels == sfh_params.shape[0]), 'Number of parameter sets must be consistent between power law and SFH.'

        ob = self._check_bounds(params)
        if(np.any(ob)):
            raise ValueError('Given parameters are out of bounds for this model (%s).' % (self.model_name))

        # This is the mean count-rate density, counts s-1 Hz-1
        countrate = self.get_model_countrate(params, stellar_model, stellar_params, sfh, sfh_params, exptau=exptau)

        # The counts are the width of the bandpass times the mean countrate density
        # times the exposure time. The definition of the "width" of an arbitrary bandpass
        # is not terribly important, since the countrate spectrum is not defined outside of
        # where we define the ARF.
        counts = np.zeros((Nmodels, self.Nfilters))
        for i, filter_label in enumerate(self.filters):
            nu_max = np.amax(self.nu_grid_obs[np.nonzero(self.filters[filter_label])])
            nu_min = np.amin(self.nu_grid_obs[np.nonzero(self.filters[filter_label])])

            counts[:,i] = self.exposure[i] * (nu_max - nu_min) * countrate[:,i]

        return counts
