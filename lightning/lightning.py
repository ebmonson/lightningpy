#!/usr/bin/env python

'''
    lightning.py

    Define an object-oriented interface for calling lightning.

    TODO:
    - Add optical-IR AGN model
    - Whole X-ray thing
    - Update and homogenize documentation
'''

# Standard library
import time
import warnings
# Scipy/numpy
import numpy as np
from scipy.integrate import trapz
from scipy.interpolate import interp1d
# Astropy
from astropy.cosmology import FLRW, FlatLambdaCDM
import astropy.units as u
import astropy.constants as const
from astropy.utils.exceptions import AstropyUserWarning
from astropy.table import Table
from astropy.io import ascii
# Lightning
from .sfh import PiecewiseConstSFH
from .sfh.delayed_exponential import DelayedExponentialSFH
from .stellar import StellarModel
from .dust import DustModel
from .attenuation.calzetti import CalzettiAtten, ModifiedCalzettiAtten
from .get_filters import get_filters

__all__ = ['Lightning']

class Lightning:
    '''
        An interface for estimating the likelihood of an SED model.
        Holds information about the filters, observed flux, type of model.
        The goal is to have one object for an entire fitting process and to
        call some kind of method, like Lightning.fit(p0, method='mcmc') to produce
        the best fit or sample the distributions.

        Attributes:
            flux_obs: numpy ndarray, (Nfilters,), float32
                Observed flux in ... units
            flux_obs_unc: numpy ndarray, (Nfilters,), float32
                Error in observed flux in ... units
            wave_grid: numpy ndarray, (npoints,), float32
                Wavelength grid on which model is evaluated.
                Calculations are done with wavelength in microns.
            filter_labels: list, str
                List of filter labels.
            filters: dict, len=Nfilters, float32
                A dict of numpy float32 arrays. The keys are *filter_labels* and the
                values correspond to the normalized transmission evaluated on
                *wave_grid*.
            wave_obs: numpy ndarray, (Nfilters,), float32
                Mean wavelength of the the supplied filters.
            path_to_filters: str
                The path (relative or absolute, should just make it absolute) to the
                top filter directory.
    '''

    def __init__(self, filter_labels,
                 redshift=None, lum_dist=None,
                 flux_obs=None, flux_obs_unc=None,
                 wave_grid=(0.1, 1000, 1200),
                 SFH_type='Piecewise-Constant', ages=None,
                 atten_type='Modified-Calzetti',
                 dust_emission=False,
                 lightning_filter_path=None,
                 print_setup_time=False,
                 cosmology=None):

        self.filter_labels = filter_labels
        self.path_to_filters = lightning_filter_path

        # Cosmology
        if (cosmology is None):
            self.cosmology = FlatLambdaCDM(H0=70, Om0=0.3)
        else:
            assert (isinstance(cosmology, FlatLambdaCDM)), "'cosmology' should be provided as an astropy cosmology object"
            self.cosmology = cosmology

        # Handle distance indicators -- redshift or luminosity distance
        if (redshift is not None):
            self.redshift = redshift
            DL = self.cosmology.luminosity_distance(self.redshift).value

            if (DL < 10):
                warnings.warn('Redshift results in a luminosity distance less than 10 Mpc. Setting DL to 10 Mpc to convert flux/uncertainty to Lnu.',
                               AstropyUserWarning)
                DL = 10

        else:
            if (lum_dist is not None):

                DL = lum_dist
                # For galaxies where a user might specify DL instead of z,
                # z should be effectively 0.
                self.redshift = 0.0

            else:

                raise ValueError("At least one of 'redshift' and 'lumin_dist' must be set.")

        self.DL = DL


        # Handle fluxes if they're provided at this stage
        if (flux_obs is None):
            self._flux_obs = None
            self.Lnu_obs = None
        else:
            self.flux_obs = flux_obs

        if (flux_obs_unc is None):
            self._flux_unc = None
            self.Lnu_unc = None
        else:
            self.flux_unc = flux_obs_unc

        # Initialize wavelength grid
        # Some error handling here would be nice
        if(isinstance(wave_grid, tuple)):
            self.wave_grid_rest = np.logspace(np.log10(wave_grid[0]), np.log10(wave_grid[1]), wave_grid[2])
        else:
            self.wave_grid_rest = wave_grid

        self.wave_grid_obs = (1 + self.redshift) * self.wave_grid_rest

        self.nu_grid_rest = const.c.to(u.um / u.s).value / self.wave_grid_rest
        self.nu_grid_obs = const.c.to(u.um / u.s).value / self.wave_grid_obs

        # Initialize a dict for the filters. Might change this to a structured numpy array.
        self.filters = dict()
        for name in self.filter_labels:
            self.filters[name] = np.zeros(len(self.wave_grid_rest), dtype='float')

        # Load the filters
        t0 = time.time()
        self._get_filters()
        self.Nfilters = len(self.filters)
        t1 = time.time()

        # Get the mean wavelengths of the filters
        self.wave_obs = np.zeros(len(self.filter_labels), dtype='float')
        self._get_wave_obs()
        self.nu_obs = const.c.to(u.um / u.s).value / self.wave_obs
        t2 = time.time()

        # This block steps up the star formation history and
        # stellar population models
        allowed_SFHs = ['Piecewise-Constant', 'Delayed-Exponential']
        if SFH_type not in allowed_SFHs:
            print('Allowed SFHs are:', allowed_SFHs)
            raise ValueError("SFH type '%s' not understood." % (SFH_type))
        else:
            self.SFH_type = SFH_type

        if (self.SFH_type == 'Piecewise-Constant'):
            # Define default stellar age bins
            univ_age = self.cosmology.age(self.redshift).value
            if (ages is None):
                if (univ_age < 1):
                    raise ValueError('Redshift too large; fewer than 4 age bins in SFH.')
                elif (univ_age < 5):
                    self.ages = np.array([0, 1.0e7, 1.0e8, 1.0e9, univ_age * 1e9])
                else:
                    self.ages = np.array([0, 1.0e7, 1.0e8, 1.0e9, 5.0e9, univ_age * 1e9])
            else:
                self.ages = np.array(ages)


            self.ages = np.sort(self.ages)
            if (np.any(self.ages > univ_age * 1e9)):
                raise ValueError('None of the provided age bins can exceed the age of the Universe at z.')

            self.Nages = len(self.ages) - 1

        else:
            univ_age = self.cosmology.age(self.redshift).value
            if (ages is None):
                self.ages = np.logspace(6, np.log10(univ_age * 1e9), 20) # Log spaced grid from 1 Myr to the age of the Universe.
            else:
                self.ages = np.array(ages)

            self.ages = np.sort(self.ages)
            if (np.any(self.ages > univ_age * 1e9)):
                raise ValueError('None of the provided ages can exceed the age of the Universe at z.')

            self.Nages = len(self.ages)


        self._setup_stellar()
        t3 = time.time()

        # Set up dust attenuation
        allowed_atten = ['Calzetti', 'Modified-Calzetti', 'None', None]
        if atten_type not in allowed_atten:
            print('Allowed attenuation curves are:', allowed_atten)
            raise ValueError("Attenuation type '%s' not understood." % (atten_type))
        else:
            self.atten_type = atten_type
            self._setup_dust_att()

        t4 = time.time()

        # Set up dust emission
        if(dust_emission):
            self._setup_dust_em()
        else:
            self.dust = None

        t5 = time.time()

        # For later use, make an array of the model components and
        # figure out how many components our total model has
        self.model_components = [self.sfh, self.atten, self.dust]
        self.Nparams = 0
        for mod in self.model_components:
            if (mod is not None):
                self.Nparams += mod.Nparams

        if (print_setup_time):
            print('%.3f s elapsed in _get_filters' % (t1 - t0))
            print('%.3f s elapsed in _get_wave_obs' % (t2 - t1))
            print('%.3f s elapsed in stellar model setup' % (t3 - t2))
            print('%.3f s elapsed in dust attenuation model setup' % (t4 - t3))
            print('%.3f s elapsed in dust emission model setup' % (t5 - t4))
            print('%.3f s elapsed total' % (t5 - t0))


    @property
    def flux_obs(self):
        return self._flux_obs


    @flux_obs.setter
    def flux_obs(self, flux):
        flux = np.array(flux)
        flux_shape = flux.shape
        if (len(flux_shape) > 1):
            # Then the second column should be the uncertainties
            if (flux_shape[1] != 2):
                raise ValueError('If flux is 2D it should have shape (Nfilters,2) where the second column contains the uncertainties.')

            if (flux_shape[0] != len(self.filter_labels)):
                raise ValueError('Number of observed fluxes (%d) must correspond to number of filters in model (%d).' % (flux_shape[0], len(self.filter_labels)))

            self._flux_obs = flux[:,0]
            self.flux_unc = flux[:,1]
        else:
            if (len(flux) != len(self.filter_labels)):
                raise ValueError('Number of observed fluxes (%d) must correspond to number of filters in model (%d).' % (len(flux), len(self.filter_labels)))

            self._flux_obs = flux

        self.Lnu_obs = self._fnu_to_Lnu(self.flux_obs)


    @property
    def flux_unc(self):
        return self._flux_unc


    @flux_unc.setter
    def flux_unc(self, flux_unc):
        flux_unc = np.array(flux_unc)
        if (len(flux_unc) != len(self.filter_labels)):
            raise ValueError('Number of flux uncertainties (%d) must correspond to number of filters in model (%d).' % (len(flux_unc), len(self.filter_labels)))

        self._flux_unc = flux_unc
        self.Lnu_unc = self._fnu_to_Lnu(self.flux_unc)


    def _fnu_to_Lnu(self, flux):
        '''
            Convert fnu in mJy to Lnu in Lsun Hz-1
        '''

        Mpc_to_cm = u.Mpc.to(u.cm)
        mJy_to_cgs = 1e-26
        cgs_to_Lsun = (u.erg / u.s /u.Hz).to(u.solLum / u.Hz)

        # DL = self.cosmology.luminosity_distance(self.redshift).value
        #
        # if (DL < 10):
        #     warnings.warn('Redshift results in a luminosity distance less than 10 Mpc. Setting DL to 10 Mpc to convert flux/uncertainty to Lnu.',
        #                    AstropyUserWarning)
        #     DL = 10

        FtoL = 4 * np.pi * (self.DL * Mpc_to_cm) ** 2 * mJy_to_cgs * cgs_to_Lsun

        return FtoL * flux


    def _get_filters(self):
        '''
            Load the filters.
        '''

        self.filters = get_filters(self.filter_labels, self.wave_grid_obs, self.path_to_filters)


    def _get_wave_obs(self):
        '''
            Compute the mean wavelength of the normalized filters.
        '''

        for i, label in enumerate(self.filter_labels):
            lam = trapz(self.wave_grid_obs * self.filters[label], self.wave_grid_obs)
            self.wave_obs[i] = lam


    def _setup_stellar(self):
        '''
            Initialize SFH model and stellar population.
        '''

        if (self.SFH_type == 'Piecewise-Constant'):
            self.sfh = PiecewiseConstSFH(self.ages)
            step=True
        elif (self.SFH_type == 'Delayed-Exponential'):
            self.sfh = DelayedExponentialSFH(self.ages)
            step=False
        else:
            raise ValueError("SFH type (%s) not understood." % (self.SFH_type))

        self.stars = StellarModel(self.filter_labels, self.redshift, age=self.ages,
                                  Z_met=0.020, step=step, wave_grid=self.wave_grid_rest)


    def _setup_dust_att(self):
        '''
            Initialize dust attenuation model.
        '''

        if (self.atten_type == 'Calzetti'):
            self.atten = CalzettiAtten(self.wave_grid_rest)
        elif (self.atten_type == 'Modified-Calzetti'):
            self.atten = ModifiedCalzettiAtten(self.wave_grid_rest)
        elif ((self.atten_type is None) or (self.atten_type == 'None')):
            self.atten = None
        else:
            raise ValueError("Atten type (%s) not understood." % (self.atten_type))

    def _setup_dust_em(self):
        '''
            Initialize dust emission model.
        '''

        self.dust = DustModel(self.filter_labels, self.redshift)

    def print_params(self, verbose=False):
        '''
            If `verbose`, print a nicely formatted table
            of the models, their parameters, and the description
            of the parameters.
            Otherwise, just print the names of the parameters.
        '''

        if (verbose):
            for mod in self.model_components:
                if (mod is not None):
                    print('')
                    print('============================')
                    print(mod.model_name)
                    print('============================')
                    mod_table = Table()
                    mod_table['Parameter'] = mod.param_names
                    mod_table['Lo'] = mod.param_bounds[:,0]
                    mod_table['Hi'] = mod.param_bounds[:,1]
                    mod_table['Description'] = mod.param_descr
                    #print(mod_table)
                    ascii.write(mod_table, format='fixed_width_two_line')
        else:
            for mod in self.model_components:
                if (mod is not None):
                    print(mod.param_names)

        print('')
        print('Total parameters: %d' % (self.Nparams))


    def get_model_lnu_hires(self, params, stepwise=False):
        '''
            Construct the high-resolution spectral model.

            Inputs
            ------
                params : np.ndarray(Nmodels, Nparams) or np.ndarray(Nparams)
                    An array of model parameters. For purposes of vectorization
                    this can be a 2D array, where the first dimension cycles over
                    different sets of parameters.

                stepwise : bool
                    If true, this function returns the spectral model as a function
                    of stellar age.

            Returns
            -------
                lnu_processed : np.ndarray(Nmodels, Nwave) or np.ndarray(Nmodels, Nwave, Nages)
                    High resolution spectral model including the effects of ISM dust.

                lnu_intrinsic : np.ndarray(Nmodels, Nwave) or np.ndarray(Nmodels, Nwave, Nages)
                    High resolution spectral model not including the effects of ISM dust.
        '''

        #sfh_shape = sfh.shape # expecting ndarray(Nmodels, Nsteps)
        param_shape = params.shape # expecting ndarray(Nmodels, Nparams)

        # if (len(sfh_shape) == 1):
        #     sfh = sfh.reshape(1, sfh.size)
        #     sfh_shape = sfh.shape

        if (len(param_shape) == 1):
            params = params.reshape(1, params.size)
            param_shape = params.shape

        Nmodels = param_shape[0]
        #Nsteps = sfh_shape[1]
        Nparams = param_shape[1]

        #assert (Nmodels == param_shape[0]), 'Number of SFHs must correspond to number of parameter sets for vectorization.'
        assert (Nparams == self.Nparams), 'Number of provided parameters (%d) must match the total number of parameters expected by the model (%d). Check Lightning.print_params().' % (Nparams, self.Nparams)
        #assert (Nsteps == self.Nages), 'Number of steps in provided SFHs must correpond to number of stellar ages in model.'
        #assert Nparams == 3

        # Chunk up parameter array -- eventually this ought to be a dict or something
        sfh_params = params[:, 0:self.sfh.Nparams]
        i = self.sfh.Nparams
        if (self.atten is not None):
            atten_params = params[:,i:i + self.atten.Nparams]
            i += self.atten.Nparams
        if (self.dust is not None):
            dust_params = params[:,i:i + self.dust.Nparams]

        # exptau = modified_calzetti(self.wave_grid_rest, params[:,0], params[:,1], np.zeros(Nmodels)) # ndarray(Nmodels, len(self.wave_grid_rest))
        # exptau_youngest = modified_calzetti(self.wave_grid_rest, params[:,0], params[:,1], params[:,2])

        expminustau = self.atten.evaluate(atten_params)

        if (Nmodels == 1):
            expminustau = expminustau.reshape(1,-1)
            #exptau_youngest = exptau.reshape(1,-1)


        lnu_stellar_attenuated, lnu_stellar_unattenuated, L_TIR_stellar = self.stars.get_model_lnu_hires(self.sfh,
                                                                                                         sfh_params,
                                                                                                         exptau=expminustau,
                                                                                                         stepwise=False)

        lnu_intrinsic = lnu_stellar_unattenuated
        lnu_processed = lnu_stellar_attenuated

        if (self.dust is not None):
            lnu_dust, Lbol_dust = self.dust.get_model_lnu_hires(dust_params)
            # Dust model comes out at native resolution; we interpolate it to whatever our resolution is here.
            finterp_dust = interp1d(self.dust.wave_grid_obs, lnu_dust, bounds_error=False, fill_value=0)
            lnu_dust_interp = finterp_dust(self.wave_grid_obs)
            lnu_processed = lnu_processed + L_TIR_stellar[:,None] * lnu_dust_interp / Lbol_dust[:,None]

        # if stepwise:
        #     if (Nmodels == 1):
        #         steps_lnu_unattenuated = steps_lnu_unattenuated.reshape(Nsteps,-1)
        #         steps_lnu_attenuated = steps_lnu_attenuated.reshape(Nsteps,-1)
        #
        #     return steps_lnu_attenuated, steps_lnu_unattenuated
        # else:
        #     lnu_unattenuated = np.sum(steps_lnu_unattenuated, axis=1)
        #     lnu_attenuated = np.sum(steps_lnu_attenuated, axis=1)
        #
        #     if (Nmodels == 1):
        #         lnu_unattenuated = lnu_unattenuated.flatten()
        #         lnu_attenuated = lnu_attenuated.flatten()
        #
        #     return lnu_attenuated, lnu_unattenuated
        #

        if (Nmodels == 1):
            lnu_processed = lnu_processed.flatten()
            lnu_intrinsic = lnu_intrinsic.flatten()

        return lnu_processed, lnu_intrinsic


    def get_model_lnu(self, params, stepwise=False):
        '''
            Construct the low-resolution SED model.

            Inputs
            ------
                params : np.ndarray(Nmodels, Nparams) or np.ndarray(Nparams)
                    An array of model parameters. For purposes of vectorization
                    this can be a 2D array, where the first dimension cycles over
                    different sets of parameters.

                stepwise : bool
                    If true, this function returns the spectral model as a function
                    of stellar age.

            Returns
            -------
                lnu_processed : np.ndarray(Nmodels, Nwave) or np.ndarray(Nmodels, Nwave, Nages)
                    Model including the effects of ISM dust, convolved with the filters.

                lnu_intrinsic : np.ndarray(Nmodels, Nwave) or np.ndarray(Nmodels, Nwave, Nages)
                    Model not including the effects of ISM dust, convolved with the filters.
        '''

        #sfh_shape = sfh.shape # expecting ndarray(Nmodels, Nsteps)
        param_shape = params.shape # expecting ndarray(Nmodels, Nparams)

        # if (len(sfh_shape) == 1):
        #     sfh = sfh.reshape(1, sfh.size)
        #     sfh_shape = sfh.shape

        if (len(param_shape) == 1):
            params = params.reshape(1, params.size)
            param_shape = params.shape

        Nmodels = param_shape[0]
        #Nages = sfh_shape[1]
        Nparams = param_shape[1]

        #assert (Nmodels == param_shape[0]), 'Number of SFHs must correspond to number of parameter sets for vectorization.'
        #assert (Nages == self.Nages), 'Number of steps in provided SFHs must correpond to number of stellar ages in model.'
        assert (Nparams == self.Nparams), 'Number of provided parameters (%d) must match the total number of parameters expected by the model (%d). Check Lightning.print_params().' % (Nparams, self.Nparams)

        lnu_processed_hires, lnu_intrinsic_hires = self.get_model_lnu_hires(params)

        lnu_processed = np.zeros((Nmodels, self.Nfilters))
        lnu_intrinsic = np.zeros((Nmodels, self.Nfilters))

        for i, filter_label in enumerate(self.filters):
            # Recall that the filters are normalized to 1 when integrated against wave_grid_obs
            # so we can integrate with that to get the mean Lnu in each band.
            lnu_processed[:,i] = trapz(self.filters[filter_label][None,:] * lnu_processed_hires, self.wave_grid_obs, axis=-1)
            lnu_intrinsic[:,i] = trapz(self.filters[filter_label][None,:] * lnu_intrinsic_hires, self.wave_grid_obs, axis=-1)


        if (Nmodels == 1):
            lnu_processed = lnu_processed.flatten()
            lnu_intrinsic = lnu_intrinsic.flatten()

        return lnu_processed, lnu_intrinsic


    def get_model_likelihood(self, params, negative=True):
        '''
            Log-likelihood of the model under the given parameters.

            If `negative` flag is set (on by default), returns the negative log likelihood
            (i.e. chi2 / 2).
        '''

        if ((self.Lnu_obs is None) or (self.Lnu_unc is None)):
            raise AttributeError('In order to calculate model likelihood, observed flux and uncertainty must be set.')

        lnu_model, _ = self.get_model_lnu(params, stepwise=False) # ndarray(Nmodels, Nfilters)

        chi2 = np.nansum(((lnu_model - self.Lnu_obs[None,:]) / self.Lnu_unc[None,:])**2, axis=-1)

        if(negative):
            return 0.5 * chi2
        else:
            return -0.5 * chi2


    def get_model_log_prob(self, params, priors=None, negative=True, p_bound=np.inf):
        '''
            Log-probability (eventaully, including prior probability) of the model under the
            given parameters.

            If `negative` flag is set (on by default), returns the negative log probability
            (i.e. chi2 / 2 + log(prior)).

            Inputs
            ------
                params : np.ndarray(Nmodels, Nparams)
                    An array of the parameters expected by the model. See ``Lightning.print_params()`` for
                    details on the current model parameters.
                priors : list of Nparams callables
                    Priors on the parameters.
                negative : bool
                    A flag setting whether the log probability or its opposite is returned (as e.g. when using
                    a minimization method). (Default: ``True``)
                p_bound : float
                    The magnitude of the log probability for models outside of the parameter space.
                    (Default: ``np.inf``)

            Returns
            -------
                log_prob : np.ndarray(Nmodels,)
                    The log of the probability, prior * likelihood.

        '''

        if ((self.Lnu_obs is None) or (self.Lnu_unc is None)):
            raise AttributeError('In order to calculate model likelihood, observed flux and uncertainty must be set.')

        param_shape = params.shape # expecting ndarray(Nmodels, Nparams)

        if (len(param_shape) == 1):
            params = params.reshape(1, params.size)
            param_shape = params.shape

        Nmodels = param_shape[0]
        #Nsteps = sfh_shape[1]
        Nparams = param_shape[1]

        #ob_mask = np.zeros(Nmodels, dtype='bool')

        # ob_mask = np.any(sfh < 0, axis=1) # logical or across first axis
        # if(self.dust is None):
        #     par_bounds = np.array([[0.0, 3.0],
        #                            [-2.3, 0.4],
        #                            [0.0, 3.99]])
        # else:
        #     par_bounds = np.array([[0.0, 3.0],
        #                            [-2.3, 0.4],
        #                            [0.0, 3.99],
        #                            [-10.0, 4.0],
        #                            [0.1, 25.0],
        #                            [1.0e3, 3.0e5],
        #                            [0.0, 1.0],
        #                            [0.0047, 0.0458]])


        # for i, bounds in enumerate(par_bounds):
        #     if ((params[i] < min(bounds)) or (params[i] > max(bounds))):
        #         ob = True
        #
        #

        #ob_mask = ob_mask | (np.any(params < par_bounds[:,0][None,:], axis=1) | np.any(params > par_bounds[:,1][None,:], axis=1))
        ob_mask = self.check_bounds(params)
        ib_mask = np.logical_not(ob_mask)

        #log_prob = np.zeros(Nmodels)

        # Only bother doing the math if there's any math to do
        if(np.count_nonzero(ib_mask) > 0):

            log_prior = np.zeros(Nmodels)
            log_prior[ob_mask] = -1 * p_bound

            if priors is not None:
                assert (Nparams == len(priors)), "Number of priors (%d) does not match number of parameters in the model (%d)." % (len(priors), Nparams)
                prior_arr = np.zeros((np.count_nonzero(ib_mask), Nparams))
                for i,p in enumerate(priors):
                    if p is not None:
                        prior_arr[:,i] = p(params[ib_mask,i])
                log_prior[ib_mask] = np.log(np.sum(prior_arr, axis=1))

            log_like = np.zeros(Nmodels)

            log_like[ib_mask] = self.get_model_likelihood(params[ib_mask,:], negative=False)

            log_prob = log_like + log_prior

            # print('log prior:', log_prior[0])
            # print('log like:', log_like[0])

        else:

            log_prob = np.zeros(Nmodels) - p_bound

        if (negative):

            return -1 * log_prob

        else:

            return log_prob

    def check_bounds(self, params):
        '''
            Look at the parameters and make sure that none of
            them are out of bounds. As of right now, that doesn't
            mean outside of the prior range, that means outside the
            range where the model is defined or meaningful.

            This is gonna be kinda slow, I think. With more careful handling
            of the parameters it could be improved.
        '''

        param_shape = params.shape

        if (len(param_shape) == 1):
            params = params.reshape(1, params.size)
            param_shape = params.shape

        Nmodels = param_shape[0]
        Nparams = param_shape[1]
        assert (Nparams == self.Nparams), 'Number of provided parameters (%d) must match the total number of parameters expected by the model (%d). Check Lightning.print_params().' % (Nparams, self.Nparams)

        ob_mask = np.full((Nmodels, len(self.model_components)), False)

        # Chunk up parameter array -- eventually this ought to be a dict or something
        sfh_params = params[:, 0:self.sfh.Nparams]
        i = self.sfh.Nparams

        if (self.atten is not None):
            atten_params = params[:,i:i + self.atten.Nparams]
            i += self.atten.Nparams
        else:
            atten_params = None

        if (self.dust is not None):
            dust_params = params[:,i:i + self.dust.Nparams]
        else:
            dust_params = None

        p = [sfh_params, atten_params, dust_params]

        # print(sfh_params)
        # print(atten_params)
        # print(dust_params)

        #ob_idcs = (params < self.param_bounds[:,0][None,:]) | (params > self.param_bounds[:,1][None,:])

        for i in np.arange(len(self.model_components)):
            mod = self.model_components[i]
            if mod is not None:
                ob_mask[:,i] = np.any(mod.check_bounds(p[i]), axis=1)

        #print(ob_mask)

        return np.any(ob_mask, axis=1)


    def _fit_emcee(self, p0, **kwargs):

        import emcee

        # if (self.dust is None):
        #     N_dim = self.Nsteps + 3
        # else:
        #     N_dim = self.Nsteps + 8

        Ndim = self.Nparams

        priors = kwargs['priors']
        Nwalkers = kwargs['Nwalkers']
        Nsteps = kwargs['Nsteps']

        try:
            const_dim = kwargs['const_dim']
            const_dim = np.array(const_dim)
            var_dim = np.logical_not(const_dim)
            Nconst_dim = np.count_nonzero(const_dim)
        except:
            const_dim = np.zeros(Ndim, dtype='bool')
            var_dim = np.logical_not(const_dim)
            Nconst_dim = 0

        if (const_dim is not None):
            const_vals = p0[0,const_dim]
            def log_prob_func(x):
                # We have to reconstruct the constant dimensions since they aren't
                # part of x
                Nmodels = x.shape[0]  # in the first iteration emcee samples the initial log prob for every walker at once;
                xx = np.zeros((Nmodels, Ndim)) # in subsequent iterations, it does teams of N_walkers/2
                xx[:,var_dim] = x
                xx[:,const_dim] = const_vals
                lnp = self.get_model_log_prob(xx, priors=priors, negative=False)
                # print('log prob:', lnp[0])
                # print('parameters:', xx[0,:])
                #print(lnp[:5])
                #print(x.shape, lnp.shape)
                return lnp

        else:
            def log_prob_func(x):
                lnp = self.get_model_log_prob(x, priors=priors, negative=False)
                #print(lnp[:5])
                return lnp

        #log_prob_func = lambda x: self.get_model_log_prob(x[:self.Nsteps], x[self.Nsteps:], negative=False)

        #N_burn = kwargs['N_burn']

        sampler = emcee.EnsembleSampler(Nwalkers, Ndim - Nconst_dim, log_prob_func, vectorize=True)

        #post_burn = sampler.run_mcmc(p0, N_burn)

        state = sampler.run_mcmc(p0[:,var_dim], Nsteps, progress=True)

        return sampler


    def _fit_simplex(self, p0, **kwargs):

        from scipy.optimize import minimize

        N_dim = self.Nsteps + 3

        try:
            const_dim = kwargs['const_dim']
            const_dim = np.array(const_dim)
            var_dim = np.logical_not(const_dim)
            N_const_dim = np.count_nonzero(const_dim)
        except:
            const_dim = np.zeros(N_dim, dtype='bool')
            var_dim = np.logical_not(const_dim)
            N_const_dim = 0
        # end try

        if (const_dim is not None):
            const_vals = p0[const_dim]
            def log_prob_func(x):
                Nmodels = x.shape[0]  # in the first iteration emcee samples the initial log prob for every walker at once;
                xx = np.zeros(N_dim) # in subsequent iterations, it does teams of N_walkers/2
                xx[var_dim] = x
                xx[const_dim] = const_vals
                return self.get_model_log_prob(xx[:self.Nsteps], xx[self.Nsteps:], negative=True, p_bound=1e6)

        else:
            def log_prob_func(x):
                return self.get_model_log_prob(x[:self.Nsteps], x[self.Nsteps:], negative=True, p_bound=1e6)

        res = minimize(log_prob_func, p0[var_dim], method='CG',
                       options={'disp': True, 'norm' : 2,
                                'maxiter': (N_dim - N_const_dim) * 500})

        return res


    def fit(self, p0, **kwargs):

        if ((self.Lnu_obs is None) or (self.Lnu_unc is None)):
            raise AttributeError('In order to fit model, observed flux and uncertainty must be set.')

        method = kwargs['method']

        if(method == 'emcee'):
            res = self._fit_emcee(p0, **kwargs)
        elif(method == 'simplex'):
            res = self._fit_simplex(p0, **kwargs)
        else:
            ValueError('Fitting method "%s" not recognized.' % (method))

        return res
