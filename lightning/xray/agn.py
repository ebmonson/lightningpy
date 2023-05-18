import numpy as np

from scipy.integrate import trapz

from .plaw import plaw_expcut, XrayPlawExpcut

def lr17_L2(agn_model, agn_params):
    '''Convert intrinsic 2500 angstrom AGN luminosity to 2 keV luminosity.

    From Lusso and Risaliti (2017).
    '''

    # We could implement anisotropy here by setting an angle dependence.
    # However for now we'll bypass that.
    L2500_scaled = agn_model.L2500_rest[0,0] / agn_model.Lbol[0,0]

    # Scale the L2500 by the bolometric luminosity
    # and multiply by the luminosity parameter.
    L2500 = 10 ** (agn_params[:,0]) * L2500_scaled

    Lsun = 3.828e33 # erg s-1

    # This is in log(erg s-1 Hz-1)
    log_L2keV = 0.633 * np.log10(L2500 * Lsun) + 7.216

    # And now in Lsun Hz-1
    L2keV = 10 ** (log_L2keV - np.log10(Lsun))

    return L2keV


class AGNPlaw(XrayPlawExpcut):
    '''Simple model for AGN X-ray emission.

    Uses the Lusso and Risaliti (2017) relationship to
    connect the intrinsic accretion disk luminosity at 2500
    Angstroms to the X-ray luminosity at 2 keV.
    The high energy cutoff is fixed, but the photon index can
    vary.
    The model includes a parameter representing the deviation
    from the LR17 relationship.
    '''
    Nparams = 2
    model_name = 'AGN-Plaw'
    model_type = 'analytic'
    gridded = False
    param_names = ['PhoIndex', 'LR17_delta']
    param_descr = ['Photon index', 'Deviation from LR17 relationship.']
    param_bounds = np.array([[-2.0, 9.0],
                             [-0.6, 0.6]]) # Set the limits on this to 2 sigma of the intrinsic scatter

    def get_model_lnu_hires(self, params, agn_model, agn_params, exptau=None):

        param_shape = params.shape # expecting ndarray(Nmodels, Nparams)
        if (len(param_shape) == 1):
            params = params.reshape(1, params.size)
            param_shape = params.shape

        Nmodels = param_shape[0]
        Nparams_in = param_shape[1]

        assert (self.Nparams == Nparams_in), 'The %s model has %d parameters' % (self.model_name, self.Nparams)

        assert (Nmodels == agn_params.shape[0]), 'Number of parameter sets must be consistent between power law and AGN.'

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
        delta = params[:,1]
        # Fix E_cut at 300 keV
        E_cut = np.full(Nmodels, 300)

        # 2 keV luminosity
        L2keV = lr17_L2(agn_model, agn_params)

        L2keV *= 10**delta

        # Calculate norm
        norm = L2keV / (2.0 ** (1 - gamma))

        lnu = (1 + self.redshift) * plaw_expcut(self.energ_grid_rest, norm, gamma, E_cut)

        lnu[:, self.energ_grid_rest < 0.1] = 0

        lnu_abs = exptau * lnu

        if (Nmodels == 1):
            lnu = lnu.flatten()
            lnu_abs = lnu_abs.flatten()

        return lnu_abs, lnu

    def get_model_countrate_hires(self, params, agn_model, agn_params, exptau=None):

        param_shape = params.shape # expecting ndarray(Nmodels, Nparams)
        if (len(param_shape) == 1):
            params = params.reshape(1, params.size)
            param_shape = params.shape

        Nmodels = param_shape[0]
        Nparams_in = param_shape[1]

        assert (self.Nparams == Nparams_in), 'The %s model has %d parameters' % (self.model_name, self.Nparams)

        assert (Nmodels == agn_params.shape[0]), 'Number of parameter sets must be consistent between power law and AGN.'

        ob = self._check_bounds(params)
        if(np.any(ob)):
            raise ValueError('Given parameters are out of bounds for this model (%s).' % (self.model_name))

        lnu_obs, _ = self.get_model_lnu_hires(params, agn_model, agn_params, exptau=exptau)

        countrate = np.log10(1 / (4 * np.pi)) - 2 * np.log10(self._DL_cm) + \
                    np.log10(lnu_obs) + np.log10(self.specresp) - np.log10(self.phot_energ)

        countrate = 10 ** countrate

        # print(np.count_nonzero(countrate))

        if (Nmodels == 1):
            countrate = countrate.flatten()

        return countrate

    def get_model_lnu(self, params, agn_model, agn_params, exptau=None):

        param_shape = params.shape # expecting ndarray(Nmodels, Nparams)
        if (len(param_shape) == 1):
            params = params.reshape(1, params.size)
            param_shape = params.shape

        Nmodels = param_shape[0]
        Nparams_in = param_shape[1]

        assert (self.Nparams == Nparams_in), 'The %s model has %d parameters' % (self.model_name, self.Nparams)

        assert (Nmodels == agn_params.shape[0]), 'Number of parameter sets must be consistent between power law and AGN.'

        ob = self._check_bounds(params)
        if(np.any(ob)):
            raise ValueError('Given parameters are out of bounds for this model (%s).' % (self.model_name))

        lnu_obs, lnu_intr = self.get_model_lnu_hires(params, agn_model, agn_params, exptau=exptau)

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

    def get_model_countrate(self, params, agn_model, agn_params, exptau=None):

        param_shape = params.shape # expecting ndarray(Nmodels, Nparams)
        if (len(param_shape) == 1):
            params = params.reshape(1, params.size)
            param_shape = params.shape

        Nmodels = param_shape[0]
        Nparams_in = param_shape[1]

        assert (self.Nparams == Nparams_in), 'The %s model has %d parameters' % (self.model_name, self.Nparams)

        assert (Nmodels == agn_params.shape[0]), 'Number of parameter sets must be consistent between power law and AGN.'

        ob = self._check_bounds(params)
        if(np.any(ob)):
            raise ValueError('Given parameters are out of bounds for this model (%s).' % (self.model_name))

        countrate_hires = self.get_model_countrate_hires(params, agn_model, agn_params, exptau=exptau)

        if (Nmodels == 1):
            lnu_hires = lnu_hires.reshape(1,-1)

        countrate = np.zeros((Nmodels, self.Nfilters))

        for i, filter_label in enumerate(self.filters):
            # Recall that the filters are normalized to 1 when integrated against wave_grid_obs
            # so we can integrate with that to get the mean countrate in each band.
            countrate[:,i] = trapz(self.filters[filter_label][None,:] * countrate_hires, self.wave_grid_obs, axis=1)

        return countrate

    def get_model_counts(self, params, agn_model, agn_params, exptau=None):

        param_shape = params.shape # expecting ndarray(Nmodels, Nparams)
        if (len(param_shape) == 1):
            params = params.reshape(1, params.size)
            param_shape = params.shape

        Nmodels = param_shape[0]
        Nparams_in = param_shape[1]

        assert (self.Nparams == Nparams_in), 'The %s model has %d parameters' % (self.model_name, self.Nparams)

        assert (Nmodels == agn_params.shape[0]), 'Number of parameter sets must be consistent between power law and AGN.'

        ob = self._check_bounds(params)
        if(np.any(ob)):
            raise ValueError('Given parameters are out of bounds for this model (%s).' % (self.model_name))

        # This is the mean count-rate density, counts s-1 Hz-1
        countrate = self.get_model_countrate(params, agn_model, agn_params, exptau=None)

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
