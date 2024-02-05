'''
    stellar.py

    Stellar population modeling.
'''

import numpy as np

import astropy.units as u
import astropy.constants as const

from scipy.interpolate import interp1d
from scipy.integrate import trapz
from scipy.io import readsav

from ..base import BaseEmissionModel

__all__ = ['PEGASEModel']

#################################
# Build stellar pop'ns
# from PEGASE
#################################
class PEGASEModel(BaseEmissionModel):
    '''Stellar emission models generated using Pégase.

    These models are either:

        - A single burst of star formation at 1 solar mass yr-1, evaluated
          on a grid of specified ages
        - A binned stellar population, representing a constant epoch of star
          formation in a specified set of stellar age bins. These models are
          integrated from the above.

    Nebular extinction + continuum are included by default, lines are optional,
    added by hand.

    Parameters
    ----------
    filter_labels : list, str
        List of filter labels.
    redshift : float
        Redshift of the model.
    age : np.ndarray, (Nsteps + 1,) or (Nages,), float
        Either the bounds of ``Nsteps`` age bins for a piecewise-constant
        SFH model, or ``Nages`` stellar ages for a continuous SFH model.
    step : bool
        If ``True``, ``age`` is interpreted as age bounds for a piecewise-constant
        SFH model. Otherwise ``age`` is interpreted as the age grid for a continuous SFH.
    Z_met : {0.001, 0.004, 0.008, 0.020, 0.050, 0.100}
        The metallicity of the stellar model. For the Pégase models, only the above metallicities
        are available.
    add_lines : bool
        If ``True``, emission lines are added to the spectral models at ages ``< 1e7`` yr.
    wave_grid : np.ndarray, (Nwave,), float, optional
        If set, the spectra are interpreted to this wavelength grid.

    Attributes
    ----------
    filter_labels
    redshift
    age
    step
    metallicity
    mstar : np.ndarray, (Nages,), float
        Surviving stellar mass as a function of age.
    Lbol : np.ndarray, (Nages,), float
        Bolometric luminosity as a function of age.
    q0 : np.ndarray, (Nages,), float
        Lyman-continuum photon production rate (yr-1) as a function of age.
    wave_grid_rest : np.ndarray, (Nwave,), float
        Rest-frame wavelength grid.
    wave_grid_obs
    nu_grid_rest
    nu_grid_obs
    Lnu_obs : np.ndarray, (Nages, Nwave), float
        ``(1 + redshift)`` times the rest-frame spectral model,
        as a function of age.

    '''

    model_name = 'Pegase-Stellar'
    model_type = 'Stellar-Emission'
    gridded = False

    Nparams = 1
    param_names = ['Zmet']
    param_descr = ['Metallicity (mass fraction, where solar = 0.020)']
    param_names_fncy = [r'$Z$']
    param_bounds = np.array([0.001, 0.100]).reshape(1,2)

    def _construct_model(self, age=None, step=True, Z_met=0.020, wave_grid=None, cosmology=None, nebular_effects=True):
        '''
            Load the appropriate models from the IDL files Rafael creates and either integrate
            them in bins (if ``step==True``) or interpolate them to an age grid otherwise.
        '''

        Zgrid = [0.001, 0.004, 0.008, 0.013, 0.016, 0.020, 0.050, 0.100]
        Nmet = len(Zgrid)
        self.Zmet = Zgrid

        if (nebular_effects):
            self.path_to_models = [self.path_to_models + '04-single_burst/Kroupa01/' + 'Kroupa01_Z%5.3f_nebular_spec.idl' % (Z_met) for Z_met in Zgrid]
        else:
            self.path_to_models = [self.path_to_models + '04-single_burst/Kroupa01/' + 'Kroupa01_Z%5.3f_spec.idl' % (Z_met) for Z_met in Zgrid]

        burst_dict = readsav(self.path_to_models[0]) # Read in the first file to get the ages, wavelength, etc.

        if cosmology is None:
            from astropy.cosmology import FlatLambdaCDM
            cosmology = FlatLambdaCDM(H0=70, Om0=0.3)

        univ_age = cosmology.age(self.redshift).value * 1e9

        if (age is None):
            if (not step):
                #raise ValueError('Ages of stellar models must be specified.')
                # Truncate gridded ages to age of Universe.

                self.age = np.array(list(burst_dict['time'][burst_dict['time'] <= univ_age]) + [univ_age])
            else:
                raise ValueError('For piecewise SFH, age bins for stellar models must be specified.')
        else:
            self.age = age

        assert (~np.any(self.age > univ_age)), 'The provided ages cannot exceed the age of the Universe at z.'

        self.step = step
        self.metallicity = Z_met

        # These are views into the original dict and so
        # are read-only.
        wave_model = burst_dict['wave'] # Wavelength grid, rest frame
        nu_model = burst_dict['nu'] # Freq grid, rest frame
        time = burst_dict['time'] # Time grid

        lnu_model = np.zeros((Nmet, len(wave_model), len(time)))
        mstar = np.zeros((Nmet, len(time)))
        q0 = np.zeros((Nmet, len(time)))
        lbol = np.zeros((Nmet, len(time)))
        lnu_lines = np.zeros_like(lnu_model)

        dv = 50 * u.km / u.s
        zlines = (dv / const.c.to(u.km / u.s)).value

        lnu_model[0,:,:] = burst_dict['lnu']
        mstar[0,:] = burst_dict['mstars']
        q0[0,:] = burst_dict['nlyc']
        lbol[0,:] = burst_dict['lbol']
        if nebular_effects:
            line_idcs = {'Ly_A': 29,
                         'OIII1666':46,
                         'CIII1909': 38,
                         'OII3726':44,
                         'Hbeta':0,
                         'OIII5007': 48,
                         'Halpha':1,
                         'NII6584':42,
                         'SII6716':53,
                         'SII6730':53,
                         'SIII9068':-4,
                         'HeI1.083':33}
            self.nebular = True
            Nlines = len(line_idcs)
            l_lines_strong = np.zeros((Nmet, Nlines, len(time)))
            wave_lines = burst_dict['wlines'] # Wavelength of lines (Nlines,)

            lnu_lines_tmp = np.exp(-1.0 * ((wave_model[:,None] / wave_lines[None,:] - 1) / zlines) ** 2) # (Nwave, Nlines)
            lnu_lines_tmp /= np.abs(trapz(lnu_lines_tmp, nu_model, axis=0)) # (Nwave, Nlines)
            lnu_lines[0,:,:] = np.sum(burst_dict['l_lines'][None, :, :] * lnu_lines_tmp[:,:,None], axis=1) # Expand to (Nwave, Nlines, Nage), then sum over lines.
            l_lines_strong[0,:,:] = burst_dict['l_lines'][np.array(list(line_idcs.values())),:]
            wave_lines_strong = np.array([wave_lines[idx] for idx in line_idcs.values()])
            line_names_strong = np.array(list(line_idcs.keys()))

        for i in np.arange(1, len(self.path_to_models)):

            burst_dict = readsav(self.path_to_models[i])

            lnu_model[i,:,:] = burst_dict['lnu']
            mstar[i,:] = burst_dict['mstars']
            q0[i,:] = burst_dict['nlyc']
            lbol[i,:] = burst_dict['lbol']
            if nebular_effects:
                lnu_lines_tmp = np.exp(-1.0 * ((wave_model[:,None] / wave_lines[None,:] - 1) / zlines) ** 2) # (Nwave, Nlines)
                # In practice this normalization should be sqrt(pi) * lambda * z
                lnu_lines_tmp /= np.abs(trapz(lnu_lines_tmp, nu_model, axis=0)) # (Nwave, Nlines)
                lnu_lines[i,:,:] = np.sum(burst_dict['l_lines'][None, :, :] * lnu_lines_tmp[:,:,None], axis=1)
                l_lines_strong[i,:,:] = burst_dict['l_lines'][np.array(list(line_idcs.values())),:]

        lnu_model += lnu_lines
            # for j,t in enumerate(time):
            #     lnu_lines = np.zeros(len(burst_dict['wave']))
            #     # For each line...
            #     for i,wave in enumerate(wave_lines):
            #         lnu_line = np.exp(-1.0 * (((wave_model / wave) - 1) / z)**2)
            #         lnu_line = lnu_line / np.abs(trapz(lnu_line, nu_model))
            #         lnu_lines = lnu_lines + l_lines[i,j] * lnu_line
            #
            #     lnu_model[:,j] = lnu_model[:,j] + lnu_lines

            # lnu_line = np.exp(-1.0 * (((wave_model / wave_lines) - 1) / z)**2)
            #
            # lnu_lines[0,:,:]

        # lnu_model = np.array(burst_dict['lnu']) # Lnu[wave, time] (rest frame)
        # # wave_lines = burst_dict['wlines'] # Wavelength of lines
        # # l_lines = burst_dict['l_lines'] # Integrated luminosity of lines
        #
        # mstar = burst_dict['mstars'] # Stellar mass
        # q0 = burst_dict['nlyc'] # Number of lyman continuum photons
        # lbol = burst_dict['lbol'] # Bolometric luminosity

        wave_model_obs = wave_model * (1 + self.redshift)
        nu_model_obs = nu_model / (1 + self.redshift)

        # Handle emission lines: Add a narrow gaussian
        # to the model at the location of each line


        # if(nebular_effects):
        #     self.nebular = True
        #     # Line names aren't given, so we have to
        #     # establish them here. We're missing HeII lines,
        #     # notabley, along with CIII and SIII
        #     line_idcs = {'Ly_A': 29,
        #                  'OIII1666':46,
        #                  'CIII1909': 38,
        #                  'OII3726':44,
        #                  'Hbeta':0,
        #                  'OIII5007': 48,
        #                  'Halpha':1,
        #                  'NII6584':42,
        #                  'SII6716':53,
        #                  'SII6730':53,
        #                  'SIII9068':-4,
        #                  'HeI1.083':33
        #                  }
        #
        #
        #     Nlines = len(line_idcs)
        #     wave_lines = burst_dict['wlines'] # Wavelength of lines (Nlines,)
        #     l_lines = burst_dict['l_lines'] # Integrated luminosity of lines (Nlines, Nages)
        #     wave_lines_strong = np.array([wave_lines[idx] for idx in line_idcs.values()])
        #     l_lines_strong = np.array([l_lines[idx,:] for idx in line_idcs.values()])
        #     line_names_strong = list(line_idcs.keys())
        #
        #     # For each timestep...
        #     for j,t in enumerate(time):
        #         lnu_lines = np.zeros(len(burst_dict['wave']))
        #         # For each line...
        #         for i,wave in enumerate(wave_lines):
        #             lnu_line = np.exp(-1.0 * (((wave_model / wave) - 1) / z)**2)
        #             lnu_line = lnu_line / np.abs(trapz(lnu_line, nu_model))
        #             lnu_lines = lnu_lines + l_lines[i,j] * lnu_line
        #
        #         lnu_model[:,j] = lnu_model[:,j] + lnu_lines

        lnu_obs = lnu_model * (1 + self.redshift)

        # From here:
        #  TODO: Add a new dimension to all arrays for metallicity.

        if (self.step):

            Nbins = len(age) - 1
            self.Nages = Nbins

            q0_age = np.zeros((Nbins, Nmet), dtype='double') # Ionizing photons per bin
            lnu_age = np.zeros((Nbins, Nmet, len(wave_model)), dtype='double') # Lnu(wave) per bin
            lbol_age = np.zeros((Nbins, Nmet), dtype='double') # Bolometric luminosity in bin
            mstar_age = np.zeros((Nbins, Nmet), dtype='double') # Mass in bin
            if (nebular_effects): l_lines_age = np.zeros((Nbins, Nmet, Nlines)) # Integrated line luminosities

            #dt = 5.e5 # Time resolution in years
            n_substeps = 100 # number of time divisions -- faster to do it this way, seemingly no loss of accuracy
            #t0 = time_module.time()
            for i in np.arange(Nbins):
                for j in np.arange(Nmet):

                    #if (i == Nbins - 1): dt = 1e6

                    ti = age[i]
                    tf = age[i + 1]
                    bin_width = tf - ti
                    #n_substeps = (bin_width // dt) # Number of timesteps in bin
                    dt = bin_width / n_substeps
                    time_substeps = dt * np.arange(n_substeps + 1) # "Time from the onset of SF, progressing linearly"
                    integrate_here = (time_substeps >= 0) & (time_substeps <= bin_width)
                    q0_age[i,j] = trapz(np.interp(tf - time_substeps, time, q0[j,:])[integrate_here], time_substeps[integrate_here])
                    lbol_age[i,j] = trapz(np.interp(tf - time_substeps, time, lbol[j,:])[integrate_here], time_substeps[integrate_here])
                    mstar_age[i,j] = trapz(np.interp(tf - time_substeps, time, mstar[j,:])[integrate_here], time_substeps[integrate_here])

                    # Vectorize later
                    for k in np.arange(len(nu_model_obs)):
                        lnu_age[i,j,k] = trapz(np.interp(tf - time_substeps, time, lnu_obs[j,k,:])[integrate_here], time_substeps[integrate_here])

                    if (nebular_effects):
                        for k in np.arange(Nlines):
                            l_lines_age[i,j,k] = trapz(np.interp(tf - time_substeps, time, l_lines_strong[j,k,:])[integrate_here], time_substeps[integrate_here])

        else:

            Nages = len(age)
            self.Nages = Nages

            q0_age = np.zeros((Nages, Nmet), dtype='double') # Ionizing photons per bin
            lnu_age = np.zeros((Nages, Nmet, len(wave_model)), dtype='double') # Lnu(wave) per bin
            lbol_age = np.zeros((Nages, Nmet), dtype='double') # Bolometric luminosity in bin
            mstar_age = np.zeros((Nages, Nmet), dtype='double') # Mass in bin
            lnu_age = np.zeros((Nages, Nmet, len(wave_model)), dtype='double') # Lnu(wave)
            if (nebular_effects): l_lines_age = np.zeros((Nages, Nmet, Nlines))

            #t0 = time_module.time()
            for j in np.arange(Nmet):
                q0_age[:,j] = np.interp(age, time, q0[j,:])
                lbol_age[:,j] = np.interp(age, time, lbol[j,:])
                mstar_age[:,j] = np.interp(age, time, mstar[j,:])

                # Vectorize later
                for k in np.arange(len(nu_model_obs)):
                    lnu_age[:,j,k] = np.interp(age, time, lnu_obs[k,j,:])

                if (nebular_effects):
                    for k in np.arange(Nlines):
                        l_lines_age[:,j,k] = np.interp(age, time, l_lines_strong[k,j,:])


        #t1 = time_module.time()

        self.mstar = mstar_age
        self.Lbol = lbol_age
        self.q0 = q0_age
        if (nebular_effects):
            self.line_lum = l_lines_age
            self.line_names = line_names_strong
            self.line_wave = wave_lines

        c_um = const.c.to(u.micron / u.s).value

        if (wave_grid is not None):
            finterp = interp1d(wave_model, lnu_age, bounds_error=False, fill_value=0.0, axis=-1)
            lnu_age_interp = finterp(wave_grid)
            lnu_age_interp[lnu_age_interp < 0.0] = 0.0
            nu_grid = c_um / wave_grid

            self.wave_grid_rest = wave_grid
            self.wave_grid_obs = self.wave_grid_rest * (1 + self.redshift)
            self.nu_grid_rest = nu_grid
            self.nu_grid_obs = self.nu_grid_rest * (1 + self.redshift)
            self.Lnu_obs = lnu_age_interp
        else:
            self.wave_grid_rest = wave_model
            self.wave_grid_obs = self.wave_grid_rest * (1 + self.redshift)
            self.nu_grid_rest = nu_model
            self.nu_grid_obs = self.nu_grid_rest * (1 + self.redshift)
            self.Lnu_obs = lnu_age


    def get_model_lnu_hires(self, sfh, sfh_param, params, exptau=None, exptau_youngest=None, stepwise=False):
        '''Construct the high-res stellar spectrum.

        Given a SFH instance and set of parameters, the corresponding high-resolution spectrum
        is constructed. Optionally, attenuation is applied and the attenuated power is returned.

        Parameters
        ----------
        sfh : instance of lightning.sfh.PiecewiseConstSFH or lightning.sfh.FunctionalSFH
            Star formation history model.
        sfh_params : np.ndarray, (Nmodels, Nparam) or (Nparam,), float32
            Parameters for the star formation history.
        params : None
            Empty placeholder for compatibility.
        exptau : np.ndarray, (Nmodels, Nwave) or (Nwave,), float32
            ``exp(-tau)`` as a function of wavelength. If this is 2D, the
            size of the first dimension must match the size of the first
            dimension of ``sfh_params``.
        exptau_youngest : np.ndarray, (Nmodels, Nwave) or (Nwave,), float32
            This doesn't do anything at the moment, until I figure out how
            to flexibly decide which ages to apply the birth cloud attenuation to.
        stepwise : bool
            If true, the spectrum is returned as a function of stellar age.

        Returns
        -------
        lnu_attenuated : np.ndarray, (Nmodels, Nwave), (Nmodels, Nages, Nwave), or (Nwave,), float32
            The stellar spectrum as seen after the application of the ISM dust
            attenuation model.
        lnu_unattenuated : np.ndarray, (Nmodels, Nwave), (Nmodels, Nages, Nwave), or (Nwave,), float32
            The intrinsic stellar spectrum.
        L_TIR : np.ndarray, (Nmodels,) or (Nmodels, Nages)
            The total attenuated power of the stellar population.

        '''

        # sfh_shape = sfh.shape # expecting ndarray(Nmodels, n_steps)
        if (len(sfh_param.shape) == 1):
            sfh_param = sfh_param.reshape(1, -1)

        if (self.step):
            assert (sfh.type == 'piecewise'), 'Binned stellar populations require a piecewise-defined SFH.'

        Nmodels = sfh_param.shape[0]
        #n_steps = sfh_param_shape[1]

        # Explicit dependence of the attenuation on stellar age is not currently fully implemented, but would be easy to
        # do so, if we make our attenuation model functions return arrays shaped like (Nmodels, Nages, N)
        if (exptau is None):
            exptau = np.full((Nmodels,len(self.wave_grid_rest)),1)
        else:
            assert (exptau.shape[0] == Nmodels), 'First dimension of exptau must match first dimension of SFH.'

        if (exptau_youngest is None):
            exptau_youngest = exptau
        else:
            assert (exptau.shape[0] == Nmodels), 'First dimension of exptau_youngest must match first dimension of SFH.'

        # TODO: further error checking on SFH -- the instantaneous stellar models are not meaningful
        # at ages < 1 Myr

        # ages_lnu_unattenuated = sfh[:,:,None] * self.Lnu_obs[None,:,:] # ndarray(Nmodels, n_steps, len(self.wave_grid_rest))
        # ages_lnu_attenuated = ages_lnu_unattenuated.copy()
        # ages_lnu_attenuated[:,0,:] = ages_lnu_attenuated[:,0,:] * exptau_youngest
        # ages_lnu_attenuated[:,1:,:] = ages_lnu_attenuated[:,1:,:] * exptau[:,None,:]

        # Integrate steps_lnu_unattenuated - steps_lnu_attenuated to get the dust luminosity per bin
        # Comes out negative since self.nu_grid_obs is monotonically decreasing.
        #ages_L_TIR = np.abs(trapz(ages_lnu_unattenuated - ages_lnu_attenuated, self.nu_grid_obs, axis=2))

        finterp = interp1d(self.Zmet, np.log10(self.Lnu_obs), axis=1)
        Lnu_obs = 10**finterp(params.flatten())
        Lnu_obs = np.swapaxes(Lnu_obs, 0, 1)

        # It is sometimes useful to have the spectra evaluated at each stellar age
        if stepwise:

            ages_lnu_unattenuated = sfh.multiply(sfh_param, Lnu_obs)
            ages_lnu_attenuated = ages_lnu_unattenuated.copy()
            #ages_lnu_attenuated[:,0,:] = ages_lnu_attenuated[:,0,:] * exptau_youngest
            ages_lnu_attenuated = ages_lnu_attenuated * exptau[:,None,:]
            ages_L_TIR = np.abs(trapz(ages_lnu_unattenuated - ages_lnu_attenuated, self.nu_grid_obs, axis=2))

            if (Nmodels == 1):
                ages_lnu_unattenuated = ages_lnu_unattenuated.reshape(self.Nages,-1)
                ages_lnu_attenuated = ages_lnu_attenuated.reshape(self.Nages,-1)
                ages_L_TIR = ages_L_TIR.flatten()

            return ages_lnu_attenuated, ages_lnu_unattenuated, ages_L_TIR

        else:

            # We distinguish between piecewise and continuous SFHs here
            if (self.step):
                # lnu_unattenuated = np.sum(ages_lnu_unattenuated, axis=1)
                # lnu_attenuated = np.sum(ages_lnu_attenuated, axis=1)
                # L_TIR = np.sum(ages_L_TIR, axis=1)

                lnu_unattenuated = sfh.sum(sfh_param, Lnu_obs)
                # lnu_attenuated = lnu_unattenuated.copy()
                # lnu_attenuated = lnu_unattenuated * exptau
                # L_TIR = np.abs(trapz(lnu_unattenuated - lnu_attenuated, self.nu_grid_obs, axis=1))

            else:
                # lnu_unattenuated = trapz(ages_lnu_unattenuated, self.age, axis=1)
                # lnu_attenuated = trapz(ages_lnu_attenuated, self.age, axis=1)
                # L_TIR = trapz(ages_L_TIR, self.age, axis=1)

                lnu_unattenuated = sfh.integrate(sfh_param, Lnu_obs)

            lnu_attenuated = lnu_unattenuated.copy()
            lnu_attenuated = lnu_unattenuated * exptau
            L_TIR = np.abs(trapz(lnu_unattenuated - lnu_attenuated, self.nu_grid_obs, axis=1))

            if (Nmodels == 1):
                lnu_unattenuated = lnu_unattenuated.flatten()
                lnu_attenuated = lnu_attenuated.flatten()
                L_TIR = L_TIR.flatten()

            return lnu_attenuated, lnu_unattenuated, L_TIR


    def get_model_lnu(self, sfh, sfh_param, params=None, exptau=None, exptau_youngest=None, stepwise=False):
        '''Construct the stellar SED as observed in the given filters.

        Given a SFH instance and set of parameters, the corresponding high-resolution spectrum
        is constructed and convolved with the filters. Optionally, attenuation is applied and
        the attenuated power is returned.

        Parameters
        ----------
        sfh : instance of lightning.sfh.PiecewiseConstSFH or lightning.sfh.FunctionalSFH
            Star formation history model.
        sfh_params : np.ndarray, (Nmodels, Nparam) or (Nparam,), float32
            Parameters for the star formation history.
        params : None
            Empty placeholder for compatibility.
        exptau : np.ndarray, (Nmodels, Nwave) or (Nwave,), float32
            ``exp(-tau)`` as a function of wavelength. If this is 2D, the
            size of the first dimension must match the size of the first
            dimension of ``sfh_params``.
        exptau_youngest : np.ndarray, (Nmodels, Nwave) or (Nwave,), float32
            This doesn't do anything at the moment, until I figure out how
            to flexibly decide which ages to apply the birth cloud attenuation to.
        stepwise : bool
            If true, the spectrum is returned as a function of stellar age.

        Returns
        -------
        lnu_attenuated : np.ndarray, (Nmodels, Nfilters), (Nmodels, Nages, Nfilters), or (Nfilters,), float32
            The stellar spectrum as seen after the application of the ISM dust
            attenuation model.
        lnu_unattenuated : np.ndarray, (Nmodels, Nfilters), (Nmodels, Nages, Nfilters), or (Nfilters,), float32
            The intrinsic stellar spectrum.
        L_TIR : np.ndarray, (Nmodels,) or (Nmodels, Nages)
            The total attenuated power of the stellar population.

        '''

        if (len(sfh_param.shape) == 1):
            sfh_param = sfh_param.reshape(1, -1)


        if (self.step):
            assert (sfh.type == 'piecewise'), 'Binned stellar populations require a piecewise-defined SFH.'

        Nmodels = sfh_param.shape[0]

        if (exptau is None):
            exptau = np.full((Nmodels,len(self.wave_grid_rest)),1)
        else:
            assert (exptau.shape[0] == Nmodels), 'First dimension of exptau must match first dimension of SFH.'


        if (exptau_youngest is None):
            exptau_youngest = exptau
        else:
            assert (exptau.shape[0] == Nmodels), 'First dimension of exptau_youngest must match first dimension of SFH.'

        # It is sometimes useful to have the spectra evaluated at each stellar age
        if stepwise:

            ages_lnu_attenuated, ages_lnu_unattenuated, ages_L_TIR = self.get_model_lnu_hires(sfh, sfh_param, exptau=exptau, exptau_youngest=exptau_youngest, stepwise=True)

            if (Nmodels == 1):
                ages_lnu_attenuated = ages_lnu_attenuated.reshape(1, self.Nages, -1)
                ages_lnu_unattenuated = ages_lnu_unattenuated.reshape(1, self.Nages, -1)

            ages_lmod_attenuated = np.zeros((Nmodels, self.Nages, self.Nfilters))
            ages_lmod_unattenuated = np.zeros((Nmodels, self.Nages, self.Nfilters))

            for i, filter_label in enumerate(self.filters):
                # Recall that the filters are normalized to 1 when integrated against wave_grid_obs
                # so we can integrate with that to get the mean Lnu in each band.
                ages_lmod_attenuated[:,:,i] = trapz(self.filters[filter_label][None,None,:] * ages_lnu_attenuated, self.wave_grid_obs, axis=2)
                ages_lmod_unattenuated[:,:,i] = trapz(self.filters[filter_label][None,None,:] * ages_lnu_unattenuated, self.wave_grid_obs, axis=2)

            if (Nmodels == 1):
                ages_lmod_attenuated = ages_lmod_attenuated.flatten()
                ages_lmod_unattenuated = ages_lmod_unattenuated.flatten()
                ages_L_TIR = ages_L_TIR.flatten()

            return ages_lmod_attenuated, ages_lmod_unattenuated, ages_L_TIR
        else:

            lnu_attenuated, lnu_unattenuated, L_TIR = self.get_model_lnu_hires(sfh, sfh_param, exptau=exptau, exptau_youngest=exptau_youngest, stepwise=False)

            if (Nmodels == 1):
                lnu_attenuated = lnu_attenuated.reshape(1,-1)
                lnu_unattenuated = lnu_unattenuated.reshape(1,-1)

            lmod_attenuated = np.zeros((Nmodels, self.Nfilters))
            lmod_unattenuated = np.zeros((Nmodels, self.Nfilters))

            for i, filter_label in enumerate(self.filters):
                # Recall that the filters are normalized to 1 when integrated against wave_grid_obs
                # so we can integrate with that to get the mean Lnu in each band.
                lmod_attenuated[:,i] = trapz(self.filters[filter_label][None,:] * lnu_attenuated, self.wave_grid_obs, axis=1)
                lmod_unattenuated[:,i] = trapz(self.filters[filter_label][None,:] * lnu_unattenuated, self.wave_grid_obs, axis=1)

            # We distinguish between piecewise and continuous SFHs here
            # if (self.step):
            #     lmod_unattenuated = np.sum(ages_lmod_unattenuated, axis=1)
            #     lmod_attenuated = np.sum(ages_lmod_attenuated, axis=1)
            #     L_TIR = np.sum(ages_L_TIR, axis=1)
            # else:
            #     lmod_unattenuated = trapz(ages_lmod_unattenuated, self.age, axis=1)
            #     lmod_attenuated = trapz(ages_lmod_attenuated, self.age, axis=1)
            #     L_TIR = trapz(ages_L_TIR, self.age, axis=1)

            if (Nmodels == 1):
                lmod_attenuated = lmod_attenuated.flatten()
                lmod_unattenuated = lmod_unattenuated.flatten()
                L_TIR = L_TIR.flatten()


            return lmod_attenuated, lmod_unattenuated, L_TIR

    def get_model_lines(self, sfh, sfh_param, params, stepwise=False):

        if (len(sfh_param.shape) == 1):
            sfh_param = sfh_param.reshape(1, -1)

        if (self.step):
            assert (sfh.type == 'piecewise'), 'Binned stellar populations require a piecewise-defined SFH.'

        Nmodels = sfh_param.shape[0]

        assert (self.nebular), 'Models were created without nebular emission; there are no lines.'

        finterp = interp1d(self.Zmet, np.log10(self.line_lum), axis=1)
        L_lines = 10**finterp(params.flatten())
        L_lines = np.swapaxes(L_lines, 0, 1)

        if stepwise:
            ages_Lmod_lines = sfh.multiply(sfh_param, L_lines)

            return ages_Lmod_lines

        else:
            if (self.step):
                Lmod_lines = sfh.sum(sfh_param, L_lines)
            else:
                Lmod_lines = sfh.integrate(sfh_param, L_lines)

            return Lmod_lines
