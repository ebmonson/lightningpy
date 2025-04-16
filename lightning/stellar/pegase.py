'''
    pegase.py

    Stellar population modeling with PEGASE, and optionally Cloudy.
'''

import h5py
import numpy as np

import astropy.units as u
import astropy.constants as const

from scipy.interpolate import interp1d, interpn
from scipy.integrate import trapezoid as trapz
from scipy.io import readsav

from ..base import BaseEmissionModel

__all__ = ['PEGASEModel', 'PEGASEModelA24']

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

    def _construct_model(self, age=None, step=True, wave_grid=None, cosmology=None, nebular_effects=True):
        '''
            Load the appropriate models from the IDL files Rafael creates and either integrate
            them in bins (if ``step==True``) or interpolate them to an age grid otherwise.
        '''

        Zgrid = [0.001, 0.004, 0.008, 0.013, 0.016, 0.020, 0.050, 0.100]
        Nmet = len(Zgrid)
        self.Zmet = Zgrid

        self.modeldir = self.modeldir.joinpath('PEGASE/legacy/Kroupa01/')

        if (nebular_effects):
            # self.path_to_models = [self.path_to_models + 'PEGASE/legacy/Kroupa01/' + 'Kroupa01_Z%5.3f_nebular_spec.idl' % (Z_met) for Z_met in Zgrid]
            fnames = ['Kroupa01_Z%5.3f_nebular_spec.idl' % (Z_met) for Z_met in Zgrid]
        else:
            # self.path_to_models = [self.path_to_models + 'PEGASE/legacy/Kroupa01/' + 'Kroupa01_Z%5.3f_spec.idl' % (Z_met) for Z_met in Zgrid]
            fnames = ['Kroupa01_Z%5.3f_spec.idl' % (Z_met) for Z_met in Zgrid]

        # burst_dict = readsav(self.path_to_models[0]) # Read in the first file to get the ages, wavelength, etc.
        burst_dict = readsav(str(self.modeldir.joinpath(fnames[0])))

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
        #self.metallicity = Z_met

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

        # for i in np.arange(1, len(self.path_to_models)):
        for i in np.arange(1, len(fnames)):

            burst_dict = readsav(str(self.modeldir.joinpath(fnames[i])))

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

            Nbins = len(self.age) - 1
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

            Nages = len(self.age)
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
                    # Dims are metallicity, wavelength, time
                    lnu_age[:,j,k] = np.interp(age, time, lnu_obs[j,k,:])

                if (nebular_effects):
                    for k in np.arange(Nlines):
                        l_lines_age[:,j,k] = np.interp(age, time, l_lines_strong[j,k,:])


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

    def get_mstar_coeff(self, Z):
        '''Return the Mstar coefficients as a function of age for a given metallicity.

        Parameters
        ----------
        Z : array-like (Nmodels,)
            Stellar metallicity

        Returns
        -------
        Mstar : (Nmodels, Nages)
            Surviving stellar mass as function of age per 1 Msun yr-1 of SFR.
        '''

        finterp_mstar = interp1d(self.Zmet, self.mstar, axis=1)
        return finterp_mstar(Z).T

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

        finterp = interp1d(self.Zmet, 
                           np.log10(self.Lnu_obs, 
                                    where=(self.Lnu_obs > 0), 
                                    out=(np.zeros_like(self.Lnu_obs) - 1000.0)), 
                           axis=1)
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


    def get_model_lnu(self, sfh, sfh_param, params, exptau=None, exptau_youngest=None, stepwise=False):
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

            ages_lnu_attenuated, ages_lnu_unattenuated, ages_L_TIR = self.get_model_lnu_hires(sfh, sfh_param, params, exptau=exptau, exptau_youngest=exptau_youngest, stepwise=True)

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

            lnu_attenuated, lnu_unattenuated, L_TIR = self.get_model_lnu_hires(sfh, sfh_param, params, exptau=exptau, exptau_youngest=exptau_youngest, stepwise=False)

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
        '''Get the integrated luminosity of all the lines available to the nebular model.

        See self.line_names for a full list of lines. In the future we'll need to redden these lines
        to compare them to the observed lines, such that our line attenuation is consistent with
        the attenuation of the stellar population model broadly.

        Parameters
        ----------
        sfh : instance of lightning.sfh.PiecewiseConstSFH or lightning.sfh.FunctionalSFH
            Star formation history model.
        sfh_params : np.ndarray, (Nmodels, Nparam) or (Nparam,), float32
            Parameters for the star formation history.
        params : np.ndarray, (Nmodels, 1) or (Nmodels,)
            Values for Z.
        stepwise : bool
            If true, the lines are returned as a function of stellar age.

        Returns
        -------
        Lmod_lines :  np.ndarray, (Nmodels, Nlines) or (Nmodels, Nages, Nlines)
            Integrated line luminosities, optionally as a function of age.

        '''

        if (len(sfh_param.shape) == 1):
            sfh_param = sfh_param.reshape(1, -1)

        if (self.step):
            assert (sfh.type == 'piecewise'), 'Binned stellar populations require a piecewise-defined SFH.'

        Nmodels = sfh_param.shape[0]

        assert (self.nebular), 'Models were created without nebular emission; there are no lines.'

        finterp = interp1d(self.Zmet, 
                           np.log10(self.line_lum,
                                    where=(self.line_lum > 0),
                                    out=(np.zeros_like(self.line_lum) - 1000.0)), 
                           axis=1)
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

class PEGASEModelA24(BaseEmissionModel):
    '''Stellar emission models generated using PEGASE, including
    nebular emission calculated with Cloudy using a custom recipe with
    an ionization bounded nebula and an open geometry. See the README in the
    models folder for an outline of the Cloudy setup and links to
    more detailed documentation.

    These models are either:

        - A single burst of star formation at 1 solar mass yr-1, evaluated
          on a grid of specified ages
        - A binned stellar population, representing a constant epoch of star
          formation in a specified set of stellar age bins. These models are
          integrated from the above.

    Nebular extinction, lines, and continuum are included by default.

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
    binaries : bool
        If ``True``, the spectra include the effects of binary stellar evolution. If ``False``, the nebular model
        cannot be applied.
    nebular_effects : bool
        If ``True``, the spectra will include nebular extinction, continua, and lines.
    line_labels : np.ndarray, (Nlines,), string, optional
        Line labels in the format used by pyCloudy. See `lightning/models/linelist_full.txt` for the
        complete list of lines in the grid and their format.
    dust_grains : bool
        If ``True``, then dust grains are included in the Cloudy grids. (Default: False)
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

    model_name = 'PEGASE-Stellar-A24'
    model_type = 'Stellar-Emission'
    gridded = False

    def _construct_model(self, age=None, lognH=2.0, step=True,
                         wave_grid=None, cosmology=None,
                         nebular_effects=True, line_labels=None, dust_grains=False):
        '''
            Load the appropriate models from the PEGASE h5 files and either integrate
            them in bins (if ``step==True``) or interpolate them to an age grid otherwise.
        '''
        # self.path_to_linelist = self.path_to_models
        self.path_to_linelist = self.modeldir

        dust_str = 'g' if dust_grains else 'ng'

        # if dust_grains:
        #     self.path_to_models = self.path_to_models + 'PEGASE/Cloudy/imf_kroupa01/' + 'PEGASE_imf_kroupa01_fullgrid_g.h5'
        # else:
        #     self.path_to_models = self.path_to_models + 'PEGASE/Cloudy/imf_kroupa01/' + 'PEGASE_imf_kroupa01_fullgrid_ng.h5'

        self.modeldir = self.modeldir.joinpath('PEGASE/Cloudy/imf_kroupa01/')
        fname = 'PEGASE_imf_kroupa01_fullgrid_%s.h5' % dust_str

        #f = h5py.File(self.path_to_models)
        f = h5py.File(self.modeldir.joinpath(fname).open('rb'))

        self.Zmet = f['Zstars'][:]

        if cosmology is None:
            from astropy.cosmology import FlatLambdaCDM
            cosmology = FlatLambdaCDM(H0=70, Om0=0.3)

        univ_age = cosmology.age(self.redshift).value * 1e9

        if (age is None):
            if (not step):
                # Truncate gridded ages to age of Universe; the oldest age will be
                # the age of the Universe
                self.age = np.array(list((f['age'][:])[f['age'][:] <= univ_age]) + [univ_age]) # Time grid
            elif (step):
                raise ValueError('For piecewise SFH, age bins for stellar models must be specified.')
        else:
            self.age = age

        assert (~np.any(self.age > univ_age)), 'The provided ages cannot exceed the age of the Universe at z.'

        self.step = step

        wave_model = f['wave'][:] # Wavelength grid, rest frame
        nu_model = f['nu'][:] # Freq grid, rest frame
        time = f['age'][:] # Time grid
        time_lo = np.concatenate([[0], time[:-1]])
        time_hi = time
        deltat = time_hi - time_lo
        logU = f['logU'][:] # Ionization parameter
        self.logU = logU
        self.nH = f['nH'][:]
        # self.line_names = f['lines/names'][:]
        line_names = np.array([s.decode() for s in f['lines/names'][:]])
        #Nlines = len(linenames)

        nhmask = np.log10(self.nH) == lognH

        if (nebular_effects):

            if (line_labels is None) or (str(line_labels) == 'default'):
                with self.path_to_linelist.joinpath('linelist_default.txt').open('rb') as lf:
                    self.line_labels = np.loadtxt(lf, dtype='<U16')
                # self.line_labels = np.loadtxt(self.path_to_linelist + 'linelist_default.txt', dtype='<U16')
            elif str(line_labels) == 'full':
                with self.path_to_linelist.joinpath('linelist_full.txt').open('rb') as lf:
                    self.line_labels = np.loadtxt(lf, dtype='<U16')
                # self.line_labels = np.loadtxt(self.path_to_linelist + 'linelist_full.txt', dtype='<U16')
            else:
                self.line_labels = line_labels

            # print(self.line_labels)

            line_idcs = np.array([np.asarray(line_names == l).nonzero()[0] for l in self.line_labels])
            assert line_idcs.size != 0, 'Specified lines are not in database. Check formatting of line labels in the full list.'

            # In the PEGASE model the ionizing flux falls
            # to exactly 0 for t > 50 Myr
            lnu_model = np.squeeze(f['spec/neb'][:,:,:,nhmask,:])
            # lineratios = f['lines/ratios'][:,:,:,:]
            linelum = np.squeeze((f['lines/lum'][:,:,:,:,:])[:,:,:,nhmask,line_idcs])
            wlines = (f['lines/wave'][:])[line_idcs]
            mstar = f['mstar'][:,:]
            q0 = 10**f['logq0'][:,:]
            lbol = f['Lbol'][:,:]

            # lineratios[np.isnan(lineratios)] = 0.0

            self.nebular = True
            self.wave_lines = wlines.flatten()
            self.Nparams = 2
            self.param_names = ['Zmet', 'logU']
            self.param_descr = ['Metallicity (mass fraction, where solar = 0.020 ~ 10**[-1.7])',
                                'log10 of the ionization parameter']
            self.param_names_fncy = [r'$Z$', r'$\log \mathcal{U}$']
            self.param_bounds = np.array([[np.min(self.Zmet), np.max(self.Zmet)],
                                          [-4, -1.5]])
        else:

            lnu_model = f['spec/noneb'][:,:,:]
            mstar = f['mstar'][:]
            q0 = 10**f['logq0'][:,:]
            lbol = f['Lbol'][:,:]

            # lineratios = np.zeros((len(time), len(self.Zmet), len(self.line_names)))

            self.nebular = False
            self.line_labels = [None]
            self.wave_lines = None
            self.Nparams = 1
            self.param_names = ['Zmet']
            self.param_descr = ['Metallicity (mass fraction, where solar = 0.020 ~ 10**[-1.7])']
            self.param_names_fncy = [r'$Z$']
            self.param_bounds = np.array([np.min(self.Zmet), np.max(self.Zmet)]).reshape(1,2)


        f.close()

        wave_model_obs = wave_model * (1 + self.redshift)
        nu_model_obs = nu_model / (1 + self.redshift)

        lnu_obs = lnu_model * (1 + self.redshift)

        if (self.step):

            Nbins = len(self.age) - 1
            dt_bins = np.array(self.age[1:]) - np.array(self.age[:-1])
            if np.any(dt_bins < 10**6.0):
                raise ValueError('The minimum age bin width is 10**6.05 years; this is set by the time resolution of the source models.')
            self.Nages = Nbins

            q0_age = np.zeros((Nbins,len(self.Zmet)), dtype='double') # Ionizing photons per bin
            if (nebular_effects):
                lnu_age = np.zeros((Nbins, len(self.Zmet), len(self.logU), len(wave_model)), dtype='double') # Lnu(wave) per bin and logU
                #lineratios_age = np.zeros((Nbins, len(self.Zmet), len(self.logU), len(self.line_labels)), dtype='double')
                linelum_age = np.zeros((Nbins, len(self.Zmet), len(self.logU), len(self.line_labels)), dtype='double')
            else:
                lnu_age = np.zeros((Nbins, len(self.Zmet), len(wave_model)), dtype='double') # Lnu(wave) per bin
                #lineratios_age = np.zeros((Nbins, len(self.Zmet), len(self.line_labels)))
                linelum_age = np.zeros((Nbins, len(self.Zmet), len(self.line_labels)))
            lbol_age = np.zeros((Nbins,len(self.Zmet)), dtype='double') # Bolometric luminosity in bin
            mstar_age = np.zeros((Nbins,len(self.Zmet)), dtype='double') # Mass in bin

            for i in np.arange(Nbins):

                #if (i == Nbins - 1): dt = 1e6

                ti = self.age[i]
                tf = self.age[i + 1]

                # tf and ti are the limits of integration, time_lo and time_hi are the limits of the source bins.
                fullbins = (time_hi < tf) & (time_lo >= ti)
                partialbinlo = ((time_lo < ti) & (time_hi >= ti))
                partialbinhi = ((time_lo < tf) & (time_hi >= tf))
                q0_age[i,:] = np.sum(q0[fullbins,:] * deltat[fullbins,None], axis=0)
                lbol_age[i,:] = np.sum(lbol[fullbins,:] * deltat[fullbins,None], axis=0)
                mstar_age[i,:] = np.sum(mstar[fullbins,:] * deltat[fullbins,None], axis=0)
                # lnu_age[i,...] = np.squeeze(np.sum(np.atleast_3d(lnu_obs[fullbins,...]) * deltat[fullbins,None,None], axis=0))
                if nebular_effects:
                    linelum_age[i,...] = np.sum(linelum[fullbins,...] * deltat[fullbins,None,None,None], axis=0)
                    lnu_age[i,...] = np.sum(lnu_obs[fullbins,...] * deltat[fullbins,None,None,None], axis=0)
                else:
                    lnu_age[i,...] = np.sum(lnu_obs[fullbins,...] * deltat[fullbins,None,None], axis=0)

                if np.any(partialbinlo):
                    deltat_partial = time_hi[partialbinlo] - ti
                    q0_age[i,:] +=  np.squeeze(q0[partialbinlo,:] * deltat_partial)
                    lbol_age[i,:] +=  np.squeeze(lbol[partialbinlo,:] * deltat_partial)
                    mstar_age[i,:] += np.squeeze(mstar[partialbinlo,:] * deltat_partial)
                    lnu_age[i,...] += np.squeeze(lnu_obs[partialbinlo,...] * deltat_partial)
                    if nebular_effects:
                        linelum_age[i,...] += np.squeeze(linelum[partialbinlo,...] * deltat_partial)
                if np.any(partialbinhi):
                    deltat_partial = tf - time_lo[partialbinhi]
                    q0_age[i,:] += np.squeeze(q0[partialbinhi,:] * deltat_partial)
                    lbol_age[i,:] += np.squeeze(lbol[partialbinhi,:] * deltat_partial)
                    mstar_age[i,:] += np.squeeze(mstar[partialbinhi,:] * deltat_partial)
                    lnu_age[i,...] += np.squeeze(lnu_obs[partialbinhi,...] * deltat_partial)
                    if nebular_effects:
                        linelum_age[i,...] += np.squeeze(linelum[partialbinhi,...] * deltat_partial)

                # Since we effectively assume Hbeta := 1 for all t, int(Hbeta, dt, ti, tf) = tf - ti
                # if nebular_effects:
                #     lineratios_age[i,...] /= (tf - ti)

        else:

            Nages = len(self.age)
            self.Nages = Nages
            q0_age = np.zeros((self.Nages, len(self.Zmet)))
            lbol_age = np.zeros((self.Nages, len(self.Zmet)))
            mstar_age = np.zeros((self.Nages, len(self.Zmet)))

            if (nebular_effects):
                lnu_age = np.zeros((self.Nages, len(self.Zmet), len(self.logU), len(wave_model)), dtype='double') # Lnu(wave) per bin and logU
                linelum_age = np.zeros((self.Nages, len(self.Zmet), len(self.logU), len(self.line_labels)), dtype='double')

            else:
                lnu_age = np.zeros((self.Nages, len(self.Zmet), len(wave_model)), dtype='double') # Lnu(wave) per bin
                linelum_age = np.zeros((self.Nages, len(self.Zmet), len(self.line_labels)))

            for j in np.arange(len(self.Zmet)):
                q0_age[:,j] = np.interp(self.age, time, q0[:,j])
                lbol_age[:,j] = np.interp(self.age, time, lbol[:,j])
                mstar_age[:,j] = np.interp(self.age, time, mstar[:,j])

            lnu_finterp = interp1d(time, lnu_obs, axis=0)
            lnu_age = lnu_finterp(self.age)

            if (nebular_effects):
                linelum_finterp = interp1d(time, linelum, axis=0)
                linelum_age = linelum_finterp(self.age)

        self.mstar = mstar_age
        self.Lbol = lbol_age
        self.q0 = q0_age
        # self.line_ratios = lineratios_age
        self.line_lum = linelum_age

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

    def get_mstar_coeff(self, Z):
        '''Return the Mstar coefficients as a function of age for a given metallicity.

        Parameters
        ----------
        Z : array-like (Nmodels,)
            Stellar metallicity

        Returns
        -------
        Mstar : (Nmodels, Nages)
            Surviving stellar mass as function of age per 1 Msun yr-1 of SFR.
        '''

        finterp_mstar = interp1d(self.Zmet, self.mstar, axis=1)
        return finterp_mstar(Z).T

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
        params : np.ndarray, (Nmodels, 1) or (Nmodels,)
            Values for logU, if the model includes a nebular component.
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

        if (self.nebular):
            assert (params is not None), 'BPASS models with the Cloudy component enabled require logU and Z to be specified.'
            assert (params.shape[0] == Nmodels), 'First dimension of stellar param array must match first dimension of SFH.'

        ob_mask = self._check_bounds(params)
        if np.any(ob_mask):
            raise ValueError('%d stellar param value(s) are out of bounds' % (np.count_nonzero(ob_mask)))

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

        if (self.nebular):
            # No boundary handling since we did it above.
            Zmet = params[:,0]
            logU = params[:,1]

            Lnu_transp = np.transpose(self.Lnu_obs, axes=[1,2,0,3])
            Lnu_obs = 10**interpn((self.Zmet, self.logU),
                                   np.log10(Lnu_transp, 
                                            where=((Lnu_transp > 0) & (np.isfinite(Lnu_transp))), 
                                            out=(np.zeros_like(Lnu_transp) - 1000.0)),
                                   params,
                                   method='linear')
            
        else:
            # We need only interpolate in the metallicity dimension.
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
                lnu_unattenuated = sfh.sum(sfh_param, Lnu_obs)

            else:

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
        params : np.ndarray, (Nmodels, 1) or (Nmodels,)
            Values for logU, if the model includes a nebular component.
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

        if (self.nebular):
            assert (params is not None), 'BPASS models with the Cloudy component enabled require logU and Z to be specified.'
            assert (params.shape[0] == Nmodels), 'First dimension of stellar param array must match first dimension of SFH.'

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

            ages_lnu_attenuated, ages_lnu_unattenuated, ages_L_TIR = self.get_model_lnu_hires(sfh, sfh_param, params=params, exptau=exptau, exptau_youngest=exptau_youngest, stepwise=True)

            if (Nmodels == 1):
                ages_lnu_attenuated = ages_lnu_attenuated.reshape(1, self.Nages, -1)
                ages_lnu_unattenuated = ages_lnu_unattenuated.reshape(1, self.Nages, -1)

            # Integrate steps_lnu_unattenuated - steps_lnu_attenuated to get the dust luminosity per bin
            # Comes out negative since self.nu_grid_obs is monotonically decreasing.

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

            lnu_attenuated, lnu_unattenuated, L_TIR = self.get_model_lnu_hires(sfh, sfh_param, params=params, exptau=exptau, exptau_youngest=exptau_youngest, stepwise=False)

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

            if (Nmodels == 1):
                lmod_attenuated = lmod_attenuated.flatten()
                lmod_unattenuated = lmod_unattenuated.flatten()
                L_TIR = L_TIR.flatten()


            return lmod_attenuated, lmod_unattenuated, L_TIR

    def get_model_lines(self, sfh, sfh_param, params, exptau=None, stepwise=False):
        '''Get the integrated luminosity of all the lines available to the nebular model.

        See self.line_names for a full list of lines. In the future we'll need to redden these lines
        to compare them to the observed lines, such that our line attenuation is consistent with
        the attenuation of the stellar population model broadly.

        Parameters
        ----------
        sfh : instance of lightning.sfh.PiecewiseConstSFH or lightning.sfh.FunctionalSFH
            Star formation history model.
        sfh_params : np.ndarray, (Nmodels, Nparam) or (Nparam,), float32
            Parameters for the star formation history.
        params : np.ndarray, (Nmodels, 2)
            Values for Z and logU.
        stepwise : bool
            If true, the lines are returned as a function of stellar age.

        Returns
        -------
        Lmod_lines :  np.ndarray, (Nmodels, Nlines) or (Nmodels, Nages, Nlines)
            Integrated line luminosities, optionally as a function of age.

        '''

        if (len(sfh_param.shape) == 1):
            sfh_param = sfh_param.reshape(1, -1)


        if (self.step):
            assert (sfh.type == 'piecewise'), 'Binned stellar populations require a piecewise-defined SFH.'

        Nmodels = sfh_param.shape[0]

        assert (self.nebular), 'Models were created without nebular emission; there are no lines.'
        assert (params.shape[0] == Nmodels), 'First dimension of stellar param array must match first dimension of SFH.'

        if (exptau is None):
            exptau = np.full((Nmodels,len(self.wave_lines)),1)
        else:
            assert (exptau.shape[0] == Nmodels), 'First dimension of exptau must match first dimension of SFH.'

        ob_mask = self._check_bounds(params)
        if np.any(ob_mask):
            raise ValueError('%d stellar param value(s) are out of bounds' % (np.count_nonzero(ob_mask)))

        lines_transp = np.transpose(self.line_lum, axes=[1,2,0,3])
        L_lines = 10**interpn((self.Zmet, self.logU),
                               np.log10(lines_transp,
                                        where=(lines_transp > 0),
                                        out=(np.zeros_like(lines_transp) - 1000.0)),
                               params,
                               method='linear')

        # Send the originally 0 luminosities made NaN by log interpolation back to 0.
        L_lines[np.isnan(L_lines)] = 0

        if stepwise:
            ages_Lmod_lines = sfh.multiply(sfh_param, L_lines)
            ages_Lmod_lines_ext = exptau[:,None,:] * ages_Lmod_lines

            return ages_Lmod_lines_ext, ages_Lmod_lines

        else:
            if (self.step):
                Lmod_lines = sfh.sum(sfh_param, L_lines)
            else:
                Lmod_lines = sfh.integrate(sfh_param, L_lines)

            Lmod_lines_ext = exptau * Lmod_lines

            return Lmod_lines_ext, Lmod_lines

class PEGASEBurstA24(PEGASEModelA24):
    '''
    SFH-free model representing a single instantaneous burst
    of star formation with a given mass and age.
    '''

    def __init__(self, filter_labels, redshift, wave_grid=None, age=None, lognH=2.0, cosmology=None,
                 line_labels=None, dust_grains=False):

        if cosmology is None:
            from astropy.cosmology import FlatLambdaCDM
            cosmology = FlatLambdaCDM(H0=70, Om0=0.3)

        univ_age = cosmology.age(redshift).value * 1e9

        super().__init__(filter_labels, redshift, step=False, wave_grid=wave_grid, age=age, lognH=lognH, cosmology=cosmology,
                         line_labels=line_labels, dust_grains=dust_grains)

        # Overwrite parameters for clarity
        self.nebular = True # Why wouldn't you
        self.Nparams = 4
        self.param_names = ['logMburst', 'logtburst', 'Zmet', 'logU']
        self.param_descr = ['Stellar mass formed, in solar masses',
                            'Age of burst in years',
                            'Metallicity (mass fraction, where solar = 0.020 ~ 10**[-1.7])',
                            'log10 of the ionization parameter']
        self.param_names_fncy = [r'$\log M_{\rm burst}$', r'$\log t_{\rm burst}$', r'$Z$', r'$\log \mathcal{U}$']
        self.param_bounds = np.array([[0, 9],
                                      [6, np.log10(univ_age)],
                                      [np.min(self.Zmet), np.max(self.Zmet)],
                                      [-4, -1.5]])

    def get_model_lnu_hires(self, params, exptau=None):
        '''

        Parameters
        ----------
        params : array-like (Nmodels, 4)
            The parameters are, in order, Mburst, tburst, Z, and logU. In practice we'll probably
            sample log(Mburst) and log(tburst).
        exptau : array-like (Nmodels, Nwave)
            Attenuation curve(s) evaluated at model wavelengths; really it's exp(-tau)

        Returns
        -------
        lnu_unattenuated
        lnu_unattenuated
        L_TIR

        '''

        params = params.copy()

        if len(params.shape) == 1:
            params = params.reshape(1,-1)

        Nmodels = params.shape[0]
        Mburst = 10**params[:,0]
        params[:,1] = 10**params[:,1]
        # And:
        # tburst = params[:,1]
        # Zmet = params[:,2]
        # logU = params[:,3]

        # finterp_lnu = interp1d(self.age, 
        #                        np.log10(self.Lnu_obs,
        #                                 where=(self.Lnu_obs > 0),
        #                                 out=(np.zeros_like(self.Lnu_obs) - 1000)), 
        #                        axis=0)

        # The axes go (age, Z, logU, wave)
        lnu_unattenuated = 10**interpn((self.age, self.Zmet, self.logU),
                                 np.log10(self.Lnu_obs,
                                        where=(self.Lnu_obs > 0),
                                        out=(np.zeros_like(self.Lnu_obs) - 1000)),
                                 params[:,1:],
                                 method='linear')

        # The source models are currently NaN at some (X-ray) wavelengths. Will
        # have to fix that.
        lnu_unattenuated[np.isnan(lnu_unattenuated)] = 0.0

        lnu_unattenuated *= Mburst[:,None]

        if (exptau is None):
            exptau = np.full((Nmodels,len(self.wave_grid_obs)),1)

        lnu_attenuated = exptau * lnu_unattenuated

        L_TIR = np.abs(trapz(lnu_unattenuated - lnu_attenuated, self.nu_grid_obs, axis=1))

        # lmod_attenuated = np.zeros((Nmodels, self.Nfilters))
        # lmod_unattenuated = np.zeros((Nmodels, self.Nfilters))
        # for i, filter_label in enumerate(self.filters):
        #     lmod_attenuated[:,i] = trapz(self.filters[filter_label][None,:] * lnu_attenuated, self.wave_grid_obs, axis=1)
        #     lmod_unattenuated[:,i] = trapz(self.filters[filter_label][None,:] * lnu_unattenuated, self.wave_grid_obs, axis=1)

        return lnu_attenuated, lnu_unattenuated, L_TIR

    def get_model_lnu(self, params, exptau=None):
        '''
        Parameters
        ----------
        params : array-like (Nmodels, 4)
            The parameters are, in order, Mburst, tburst, Z, and logU. In practice we'll probably
            sample log(Mburst) and log(tburst).
        exptau : array-like (Nmodels, Nwave)
            Attenuation curve(s) evaluated at model wavelengths; really it's exp(-tau)

        Returns
        -------
        lmod_unattenuated
        lmod_unattenuated
        L_TIR

        '''

        if (len(params.shape) == 1):
            params = params.reshape(1, -1)


        # if (self.step):
        #     assert (sfh.type == 'piecewise'), 'Binned stellar populations require a piecewise-defined SFH.'

        Nmodels = params.shape[0]

        if (self.nebular):
            assert (params is not None), 'BPASS models with the Cloudy component enabled require logU to be specified.'
            assert (params.shape[0] == Nmodels), 'First dimension of logU array must match first dimension of SFH.'

        if (exptau is None):
            exptau = np.full((Nmodels,len(self.wave_grid_rest)),1)
        else:
            assert (exptau.shape[0] == Nmodels), 'First dimension of exptau must match first dimension of SFH.'

        lnu_attenuated, lnu_unattenuated, L_TIR = self.get_model_lnu_hires(params, exptau=exptau)

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

        if (Nmodels == 1):
            lmod_attenuated = lmod_attenuated.flatten()
            lmod_unattenuated = lmod_unattenuated.flatten()
            L_TIR = L_TIR.flatten()


        return lmod_attenuated, lmod_unattenuated, L_TIR

    def get_model_lines(self, params, exptau=None):
        '''
        Parameters
        ----------
        params : array-like (Nmodels, 4)
            The parameters are, in order, Mburst, tburst, Z, and logU. In practice we'll probably
            sample log(Mburst) and log(tburst).
        exptau : array-like (Nmodels, Nwave)
            Attenuation curve(s) evaluated at model wavelengths; really it's exp(-tau). Currently unused, i.e.
            line luminosities returned are intrinsic, unreddened.

        Returns
        -------
        lmod_lines

        '''

        ob_mask = self._check_bounds(params)
        if np.any(ob_mask):
            raise ValueError('%d stellar param value(s) are out of bounds' % (np.count_nonzero(ob_mask)))

        if (len(params.shape) == 1):
            params = params.reshape(1, -1)

        params = params.copy()

        params[:,:2] = 10**params[:,:2]

        Nmodels = params.shape[0]
        Mburst = params[:,0]

        if (exptau is None):
            exptau = np.full((Nmodels,len(self.wave_lines)),1)
        else:
            assert (exptau.shape[0] == Nmodels), 'First dimension of exptau must match first dimension of SFH.'

        L_lines = 10**interpn((self.age, self.Zmet, self.logU),
                               np.log10(self.line_lum,
                                        where=(self.line_lum > 0),
                                        out=(np.zeros_like(self.line_lum) - 1000.0)),
                               params[:,1:],
                               method='linear')

        # Send the originally 0 luminosities made NaN by log interpolation back to 0.
        L_lines[np.isnan(L_lines)] = 0
        L_lines *= Mburst[:,None]

        L_lines_ext = exptau * L_lines

        return L_lines_ext, L_lines
