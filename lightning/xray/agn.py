import numpy as np

from scipy.integrate import trapz
from scipy.interpolate import interp1d, interpn
from astropy.io import fits
from astropy.table import Table
import astropy.constants as const
import astropy.units as u

from .base import XrayEmissionModel
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
                             [-np.inf, np.inf]]) # Set the limits on this to 2 sigma of the intrinsic scatter

    def get_model_lnu_hires(self, params, agn_model, agn_params, exptau=None):
        '''Construct the high-resolution spectrum in Lnu.

        This function takes in the agn model to normalize the X-ray spectrum
        with the Lusso & Risaliti (2017) L2keV - L2500 relationship.

        Parameters
        ----------
        params : np.ndarray(Nmodels, Nparams) or np.ndarray(Nparams)
            An array of model parameters. For purposes of vectorization
            this can be a 2D array, where the first dimension cycles over
            different sets of parameters.
        agn_model : lightning.agn.AGNModel
            The AGN model.
        agn_params : np.ndarray(Nmodels, Nparams_agn) or np.ndarray(Nparams_agn)
            The parameters for the AGN model.
        exptau : np.ndarray(Nmodels, Nwave) or np.ndarray(Nwave)
            The e(-tau) absorption curve.

        Outputs
        -------
        lnu_abs : np.ndarray(Nmodels, Nwave) or np.ndarray(Nwave)
            The absorbed Lnu spectrum.
        lnu_unabs : np.ndarray(Nmodels, Nwave) or np.ndarray(Nwave)
            The absorbtion-free Lnu spectrum.

        '''

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
        '''Construct the high-resolution spectrum in count-rate density.

        This function takes in the agn model to normalize the X-ray spectrum
        with the Lusso & Risaliti (2017) L2keV - L2500 relationship.

        Parameters
        ----------
        params : np.ndarray(Nmodels, Nparams) or np.ndarray(Nparams)
            An array of model parameters. For purposes of vectorization
            this can be a 2D array, where the first dimension cycles over
            different sets of parameters.
        agn_model : lightning.agn.AGNModel
            The AGN model.
        agn_params : np.ndarray(Nmodels, Nparams_agn) or np.ndarray(Nparams_agn)
            The parameters for the AGN model.
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
        '''Construct the bandpass-convolved SED in Lnu.

        This function takes in the agn model to normalize the X-ray spectrum
        with the Lusso & Risaliti (2017) L2keV - L2500 relationship.

        Parameters
        ----------
        params : np.ndarray(Nmodels, Nparams) or np.ndarray(Nparams)
            An array of model parameters. For purposes of vectorization
            this can be a 2D array, where the first dimension cycles over
            different sets of parameters.
        agn_model : lightning.agn.AGNModel
            The AGN model.
        agn_params : np.ndarray(Nmodels, Nparams_agn) or np.ndarray(Nparams_agn)
            The parameters for the AGN model.
        exptau : np.ndarray(Nmodels, Nwave) or np.ndarray(Nwave)
            The e(-tau) absorption curve.

        Outputs
        -------
        lnu_abs : np.ndarray(Nmodels, Nwave) or np.ndarray(Nwave)
            The absorbed Lnu in the bandpasses.
        lnu_unabs : np.ndarray(Nmodels, Nwave) or np.ndarray(Nwave)
            The absorbtion-free Lnu in the bandpasses.

        '''

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
        '''Construct the bandpass-convolved SED in count-rate.

        This function takes in the agn model to normalize the X-ray spectrum
        with the Lusso & Risaliti (2017) L2keV - L2500 relationship.

        Parameters
        ----------
        params : np.ndarray(Nmodels, Nparams) or np.ndarray(Nparams)
            An array of model parameters. For purposes of vectorization
            this can be a 2D array, where the first dimension cycles over
            different sets of parameters.
        agn_model : lightning.agn.AGNModel
            The AGN model.
        agn_params : np.ndarray(Nmodels, Nparams_agn) or np.ndarray(Nparams_agn)
            The parameters for the AGN model.
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
            countrate_hires = countrate_hires.reshape(1,-1)

        countrate = np.zeros((Nmodels, self.Nfilters))

        for i, filter_label in enumerate(self.filters):
            # Recall that the filters are normalized to 1 when integrated against wave_grid_obs
            # so we can integrate with that to get the mean countrate in each band.
            countrate[:,i] = trapz(self.filters[filter_label][None,:] * countrate_hires, self.wave_grid_obs, axis=1)

        return countrate

    def get_model_counts(self, params, agn_model, agn_params, exptau=None):
        '''Construct the high-resolution spectrum in Lnu.

        This function takes in the agn model to normalize the X-ray spectrum
        with the Lusso & Risaliti (2017) L2keV - L2500 relationship.

        Parameters
        ----------
        params : np.ndarray(Nmodels, Nparams) or np.ndarray(Nparams)
            An array of model parameters. For purposes of vectorization
            this can be a 2D array, where the first dimension cycles over
            different sets of parameters.
        agn_model : lightning.agn.AGNModel
            The AGN model.
        agn_params : np.ndarray(Nmodels, Nparams_agn) or np.ndarray(Nparams_agn)
            The parameters for the AGN model.
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
        countrate = self.get_model_countrate(params, agn_model, agn_params, exptau=exptau)

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

class Qsosed(XrayEmissionModel):
    '''Physically motivated quasar model for X-ray emission.

    Parametrized by the black hole mass and Eddington ratio
    (here called mdot). See Kubota & Done (2018) for details.

    The model is available in XSpec, and the implementation
    here was generated using Sherpa.

    '''
    Nparams = 2
    model_name = 'QSOSED'
    model_type = 'gridded'
    gridded = True
    param_names = ['log_M_SMBH', 'log_mdot']
    param_descr = ['log10 of the supermassive black hole mass',
                   'log10 of the Eddington ratio']
    # param_bounds = np.array([[1e5, 1e10],
    #                          [-1.5, 0.3]])
    param_bounds = np.array([[5, 10],
                             [-1.5, 0.3]])

    def _construct_model(self, wave_grid=None):

        self.path_to_models = self.path_to_models + 'xray/'

        # with fits.open(self.path_to_models + 'qsosed.fits.gz') as f:
        #
        #     models = fits.read('')

        source_table = Table.read(self.path_to_models + 'qsosed.fits.gz')

        Nmass = source_table.meta['N_MASS']
        Nmdot = source_table.meta['N_MDOT']

        # Models are gridded by increasing energy (decreasing wavelength)
        # We reverse them for consistency with the other models.
        # The references to '.data' strip away table metadata & units.
        # self.Lnu_rest = source_table['LNU'].reshape(Nmass, Nmdot, -1).data[:,:,::-1]
        Lnu_model = source_table['LNU'].reshape(Nmass, Nmdot, -1).data[:,:,::-1]
        self.L2500 = source_table['L2500'].reshape(Nmass, Nmdot).data

        wave_model = source_table['WAVE_MID'][0,::-1].data
        nu_model = source_table['NU_MID'][0,::-1].data
        energ_model = source_table['E_MID'][0,::-1].data

        self._mass_grid = source_table['MASS'].reshape(Nmass, Nmdot).data
        self._mdot_grid = source_table['LOG_MDOT'].reshape(Nmass, Nmdot).data

        self._mass_vec = self._mass_grid[:,0].flatten()
        self._mdot_vec = self._mdot_grid[0,:].flatten()

        c_um = const.c.to(u.micron / u.s).value

        if (wave_grid is not None):

            finterp = interp1d(wave_model, Lnu_model, bounds_error=False, fill_value=0.0, axis=2)
            lnu_interp = finterp(wave_grid)
            lnu_interp[lnu_interp < 0.0] = 0.0
            nu_grid = c_um / wave_grid

            self.wave_grid_rest = wave_grid
            #self.wave_grid_obs = self.wave_grid_rest * (1 + self.redshift)
            #self.nu_grid_rest = nu_grid
            #self.nu_grid_obs = self.nu_grid_rest * (1 + self.redshift)

            self.Lnu_rest = lnu_interp

        else:

            self.wave_grid_rest = wave_model
            #self.wave_grid_obs = self.wave_grid_rest * (1 + self.redshift)
            #self.nu_grid_rest = nu_model
            #self.nu_grid_obs = self.nu_grid_rest * (1 + self.redshift)
            self.Lnu_rest = Lnu_model

    def _construct_model_grid(self):
        # Not needed here
        pass

    def get_model_lnu_hires(self, params, exptau=None):
        '''Construct the high-resolution spectrum in Lnu.

        Parameters
        ----------
        params : np.ndarray(Nmodels, Nparams) or np.ndarray(Nparams)
            An array of model parameters. For purposes of vectorization
            this can be a 2D array, where the first dimension cycles over
            different sets of parameters.
        exptau : np.ndarray(Nmodels, Nwave) or np.ndarray(Nwave)
            The e(-tau) absorption curve.

        Outputs
        -------
        lnu_abs : np.ndarray(Nmodels, Nwave) or np.ndarray(Nwave)
            The absorbed Lnu spectrum.
        lnu_unabs : np.ndarray(Nmodels, Nwave) or np.ndarray(Nwave)
            The absorbtion-free Lnu spectrum.

        '''

        param_shape = params.shape # expecting ndarray(Nmodels, Nparams)
        if (len(param_shape) == 1):
            params = params.reshape(1, params.size)
            param_shape = params.shape

        Nmodels = param_shape[0]
        Nparams_in = param_shape[1]

        assert (self.Nparams == Nparams_in), 'The %s model has %d parameters' % (self.model_name, self.Nparams)

        if (exptau is None):
            exptau = np.full((Nmodels,len(self.wave_grid_rest)), 1)
        else:
            exptau_shape = exptau.shape
            assert (exptau_shape[0] == Nmodels), 'Number of parameter sets must be consistent between power law and absorption.'
            assert (exptau_shape[1] == len(self.wave_grid_rest)), 'Number of wavelength elements must be consistent between emission and absorption model.'

        ob = self._check_bounds(params)
        if(np.any(ob)):
            raise ValueError('Given parameters are out of bounds for this model (%s).' % (self.model_name))

        # params[:,0] = 10 ** params[:,0]
        mass = 10 ** params[:,0]
        mdot = params[:,1]
        params_tmp = np.stack([mass, mdot], axis=1)

        lnu = 10 ** interpn((self._mass_vec, self._mdot_vec),
                             np.log10(self.Lnu_rest),
                             params_tmp,
                             method='linear',
                             bounds_error=False,
                             fill_value=0.0)

        lnu *= (1 + self.redshift)

        lnu_abs = exptau * lnu

        return lnu_abs, lnu

    def get_model_L2500(self, params):
        '''Calculate the intrinsic L2500

        Parameters
        ----------
        params : np.ndarray(Nmodels, Nparams) or np.ndarray(Nparams)
            An array of model parameters. For purposes of vectorization
            this can be a 2D array, where the first dimension cycles over
            different sets of parameters.

        Outputs
        -------
        L2500 : np.ndarray(Nmodels,)
            Intrinsic (rest-frame) monochromatic luminosity at 2500 Angstroms, in
            Lsun Hz-1.

        '''

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

        #params[:,0] = 10 ** params[:,0]
        mass = 10 ** params[:,0]
        mdot = params[:,1]
        params_tmp = np.stack([mass, mdot], axis=1)
        #print(params_tmp.shape)

        L2500 = 10 ** interpn((self._mass_vec, self._mdot_vec),
                               np.log10(self.L2500),
                               params_tmp,
                               method='linear',
                               bounds_error=False,
                               fill_value=0.0)

        return L2500

    def get_model_countrate_hires(self, params, exptau=None):
        '''Construct the high-resolution spectrum in count-rate density.

        Parameters
        ----------
        params : np.ndarray(Nmodels, Nparams) or np.ndarray(Nparams)
            An array of model parameters. For purposes of vectorization
            this can be a 2D array, where the first dimension cycles over
            different sets of parameters.
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

        param_shape = params.shape # expecting ndarray(Nmodels, Nparams)
        if (len(param_shape) == 1):
            params = params.reshape(1, params.size)
            param_shape = params.shape

        Nmodels = param_shape[0]
        Nparams_in = param_shape[1]

        assert (self.Nparams == Nparams_in), 'The %s model has %d parameters' % (self.model_name, self.Nparams)

        # assert (Nmodels == agn_params.shape[0]), 'Number of parameter sets must be consistent between power law and AGN.'

        ob = self._check_bounds(params)
        if(np.any(ob)):
            raise ValueError('Given parameters are out of bounds for this model (%s).' % (self.model_name))

        lnu_obs, _ = self.get_model_lnu_hires(params, exptau=exptau)

        countrate = np.log10(1 / (4 * np.pi)) - 2 * np.log10(self._DL_cm) + \
                    np.log10(lnu_obs) + np.log10(self.specresp) - np.log10(self.phot_energ)

        countrate = 10 ** countrate

        # print(np.count_nonzero(countrate))

        if (Nmodels == 1):
            countrate = countrate.flatten()

        return countrate

    def get_model_lnu(self, params, exptau=None):
        '''Construct the bandpass-convolved SED in Lnu.

        Parameters
        ----------
        params : np.ndarray(Nmodels, Nparams) or np.ndarray(Nparams)
            An array of model parameters. For purposes of vectorization
            this can be a 2D array, where the first dimension cycles over
            different sets of parameters.
        exptau : np.ndarray(Nmodels, Nwave) or np.ndarray(Nwave)
            The e(-tau) absorption curve.

        Outputs
        -------
        lnu_abs : np.ndarray(Nmodels, Nwave) or np.ndarray(Nwave)
            The absorbed Lnu in the bandpasses.
        lnu_unabs : np.ndarray(Nmodels, Nwave) or np.ndarray(Nwave)
            The absorbtion-free Lnu in the bandpasses.

        '''

        param_shape = params.shape # expecting ndarray(Nmodels, Nparams)
        if (len(param_shape) == 1):
            params = params.reshape(1, params.size)
            param_shape = params.shape

        Nmodels = param_shape[0]
        Nparams_in = param_shape[1]

        assert (self.Nparams == Nparams_in), 'The %s model has %d parameters' % (self.model_name, self.Nparams)

        #assert (Nmodels == agn_params.shape[0]), 'Number of parameter sets must be consistent between power law and AGN.'

        ob = self._check_bounds(params)
        if(np.any(ob)):
            raise ValueError('Given parameters are out of bounds for this model (%s).' % (self.model_name))

        lnu_obs, lnu_intr = self.get_model_lnu_hires(params, exptau=exptau)

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

    def get_model_countrate(self, params, exptau=None):
        '''Construct the bandpass-convolved SED in count-rate.

        Parameters
        ----------
        params : np.ndarray(Nmodels, Nparams) or np.ndarray(Nparams)
            An array of model parameters. For purposes of vectorization
            this can be a 2D array, where the first dimension cycles over
            different sets of parameters.
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

        param_shape = params.shape # expecting ndarray(Nmodels, Nparams)
        if (len(param_shape) == 1):
            params = params.reshape(1, params.size)
            param_shape = params.shape

        Nmodels = param_shape[0]
        Nparams_in = param_shape[1]

        assert (self.Nparams == Nparams_in), 'The %s model has %d parameters' % (self.model_name, self.Nparams)

        #assert (Nmodels == agn_params.shape[0]), 'Number of parameter sets must be consistent between power law and AGN.'

        ob = self._check_bounds(params)
        if(np.any(ob)):
            raise ValueError('Given parameters are out of bounds for this model (%s).' % (self.model_name))

        countrate_hires = self.get_model_countrate_hires(params, exptau=exptau)

        if (Nmodels == 1):
            countrate_hires = countrate_hires.reshape(1,-1)

        countrate = np.zeros((Nmodels, self.Nfilters))

        for i, filter_label in enumerate(self.filters):
            # Recall that the filters are normalized to 1 when integrated against wave_grid_obs
            # so we can integrate with that to get the mean countrate in each band.
            countrate[:,i] = trapz(self.filters[filter_label][None,:] * countrate_hires, self.wave_grid_obs, axis=1)

        return countrate

    def get_model_counts(self, params, exptau=None):
        '''Construct the high-resolution spectrum in Lnu.

        Parameters
        ----------
        params : np.ndarray(Nmodels, Nparams) or np.ndarray(Nparams)
            An array of model parameters. For purposes of vectorization
            this can be a 2D array, where the first dimension cycles over
            different sets of parameters.
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

        param_shape = params.shape # expecting ndarray(Nmodels, Nparams)
        if (len(param_shape) == 1):
            params = params.reshape(1, params.size)
            param_shape = params.shape

        Nmodels = param_shape[0]
        Nparams_in = param_shape[1]

        assert (self.Nparams == Nparams_in), 'The %s model has %d parameters' % (self.model_name, self.Nparams)

        #assert (Nmodels == agn_params.shape[0]), 'Number of parameter sets must be consistent between power law and AGN.'

        #print(params)
        ob = self._check_bounds(params)
        if(np.any(ob)):
            raise ValueError('Given parameters are out of bounds for this model (%s).' % (self.model_name))

        # This is the mean count-rate density, counts s-1 Hz-1
        countrate = self.get_model_countrate(params, exptau=exptau)

        # The counts are the width of the bandpass times the mean countrate density
        # times the exposure time. The definition of the "width" of an arbitrary bandpass
        # is not terribly important, since the countrate spectrum is not defined outside of
        # where we define the ARF.
        counts = np.zeros((Nmodels, self.Nfilters))
        for i, filter_label in enumerate(self.filters):
            nu_max = np.amax(self.nu_grid_obs[np.nonzero(self.filters[filter_label])])
            nu_min = np.amin(self.nu_grid_obs[np.nonzero(self.filters[filter_label])])

            counts[:,i] = self.exposure * (nu_max - nu_min) * countrate[:,i]

        #print(counts[0,:])

        return counts
