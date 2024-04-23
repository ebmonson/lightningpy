'''
bpass.py

Stellar population modeling with BPASS+Cloudy.
'''

import h5py
import numpy as np

import astropy.units as u
import astropy.constants as const
from astropy.table import Table
from astropy.io import ascii

from scipy.interpolate import interp1d, interpn
from scipy.integrate import trapz
from scipy.io import readsav

from ..base import BaseEmissionModel

__all__ = ['BPASSModel', 'BPASSModelA24', 'BPASSBurstA24']

#################################
# Build stellar pop'ns
# from BPASS+Cloudy
#################################
class BPASSModel(BaseEmissionModel):
    '''Stellar emission models generated using BPASS, including
    nebular emission calculated with Cloudy. See the README in the
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
        cannot be applied (and as a result the ``dust_grains`` switch has no effect).
    nebular_effects : bool
        If ``True``, the spectra at ages <1e7.5 years will include nebular extinction, continua, and lines.
    dust_grains : bool
        If ``True``, the Cloudy models include the effects of dust grain depletion. This option has no effect
        unless ``nebular_effects`` is True. By default, we set this option to ``False``, for parity with the treatment
        of nebular emission in the Pégase stellar population models.
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

    model_name = 'BPASS-Stellar'
    model_type = 'Stellar-Emission'
    gridded = False

    # Nparams = 1
    # param_names = ['Zmet']
    # param_descr = ['Metallicity (mass fraction, where solar = 0.020)']
    # param_names_fncy = [r'$Z$']
    # param_bounds = np.array([1e-5, 0.040]).reshape(1,2)

    def _construct_model(self, age=None, step=True, Z_met=0.020,
                         wave_grid=None, cosmology=None, binaries=True,
                         nebular_effects=True, dust_grains=False):
        '''
            Load the appropriate models from the BPASS h5 hiles and either integrate
            them in bins (if ``step==True``) or interpolate them to an age grid otherwise.
        '''

        Zgrid = [1e-5, 1e-4, 0.001, 0.002, 0.003, 0.004, 0.008, 0.010, 0.014, 0.020, 0.030, 0.040]

        self.Zmet = Zgrid

        # if (Z_met not in Zgrid):
        #     print('BPASS + Cloudy models are only available for Z_met = ')
        #     print(Zgrid)
        #     raise ValueError('Z_met value not allowed.')

        # if (Z_met < 0.001):
        #     Z_str = '1em%d' % (-1 * np.log10(Z_met))
        # else:
        #     Z_str = '%03d' % (1000 * Z_met)

        Zstr = np.where(np.array(Zgrid) < 0.001, ['em%d' % (-1 * np.log10(Z)) for Z in Zgrid], ['%03d' % (1000 * Z) for Z in Zgrid])
        # print(Zstr)

        if (nebular_effects) and (not binaries):
            raise ValueError('Binaries are required to use the Cloudy nebular emission grids.')

        if (dust_grains):
            self.path_to_models = [self.path_to_models + 'BPASS_Cloudy/imf_135_300/' + 'BPASS_imf135_300_z%s_bin_gr.h5' % (s) for s in Zstr]
        else:
            self.path_to_models = [self.path_to_models + 'BPASS_Cloudy/imf_135_300/' + 'BPASS_imf135_300_z%s_bin_ng.h5' % (s) for s in Zstr]

        f = h5py.File(self.path_to_models[0])

        if cosmology is None:
            from astropy.cosmology import FlatLambdaCDM
            cosmology = FlatLambdaCDM(H0=70, Om0=0.3)

        univ_age = cosmology.age(self.redshift).value * 1e9

        if (age is None):
            if (not step):
                #raise ValueError('Ages of stellar models must be specified.')
                # Truncate gridded ages to age of Universe; the oldest age will be
                # the closest gridded age to the age of the Universe.
                self.age = np.array(list((f['age'][:])[f['age'][:] <= univ_age]) + [univ_age]) # Time grid
            elif (step):
                raise ValueError('For piecewise SFH, age bins for stellar models must be specified.')
        else:
            self.age = age

        assert (~np.any(self.age > univ_age)), 'The provided ages cannot exceed the age of the Universe at z.'

        self.step = step
        self.metallicity = Z_met

        wave_model = f['wave'][:] # Wavelength grid, rest frame
        nu_model = f['nu'][:] # Freq grid, rest frame
        time = f['age'][:] # Time grid
        deltat = np.array([10**6.05] + [10**(6.15 + 0.1 * i) - 10**(6.05 + 0.1 * i) for i in np.arange(0, len(time) - 1)])
        time_lo = np.array([0] + [10**(6.05 + 0.1 * i) for i in np.arange(0, len(time) - 1)])
        time_hi = np.array([10**6.05] + [10**(6.15 + 0.1 * i) for i in np.arange(0, len(time) - 1)])
        logU = f['logU'][:] # Ionization parameter
        self.logU = logU
        self.line_names = f['lines/names'][:]
        #Nlines = len(linenames)
        # Our convention here for ease of integration is that the wavelength is the last axis,
        # transposed from the way the models are gridded.

        f.close()

        # mstar = f['mstar'][:] # Stellar mass
        # q0 = f['q0'][:] # Number of lyman continuum photons
        # lbol = f['Lbol'][:] # Bolometric luminosity
        mstar = np.zeros((len(time), len(self.Zmet)))
        q0 = np.zeros((len(time), len(self.Zmet)))
        lbol = np.zeros((len(time), len(self.Zmet)))

        if (nebular_effects):
            lnu_model = np.zeros((len(time), len(self.Zmet), len(self.logU), len(wave_model)))
            llines = np.zeros((len(time), len(self.Zmet), len(self.logU), len(self.line_names)))
            for j in np.arange(len(self.Zmet)):
                f = h5py.File(self.path_to_models[j])
                mstar[:,j] = f['mstar'][:]
                q0[:,j] = f['q0'][:]
                lbol[:,j] = f['Lbol'][:]
                for k in np.arange(len(self.logU)):
                    lnu_model[:,j,k,:] = (f['spec_neb/%.1f' % (self.logU[k])][:,:]).T
                    linelum = (f['lines/lum/%.1f' % (self.logU[k])])[:,:]
                    llines[:linelum.shape[0],j,k,:] = linelum
                f.close()

            llines[np.isnan(llines)] = 0.0

            self.nebular = True
            self.Nparams = 2
            self.param_names = ['Zmet', 'logU']
            self.param_descr = ['Metallicity (mass fraction, where solar = 0.020)',
                                'log10 of the ionization parameter']
            self.param_names_fncy = [r'$Z$', r'$\log \mathcal{U}$']
            self.param_bounds = np.array([[1e-5, 0.040],
                                          [-4, -1]])
            # print(self.param_bounds)
        else:
            lnu_model = np.zeros((len(time), len(self.Zmet), len(wave_model)))
            for j in np.arange(len(self.Zmet)):
                f = h5py.File(self.path_to_models[j])
                mstar[:,j] = f['mstar'][:]
                q0[:,j] = f['q0'][:]
                lbol[:,j] = f['Lbol'][:]
                if (not binaries):
                    lnu_model[:,j,:] = f['spec_noneb_sin'][:,:].T
                else:
                    lnu_model[:,j,:] = f['spec_noneb'][:,:].T
                f.close()
            llines = np.zeros((len(time), len(self.Zmet), len(self.line_names)))
            self.nebular = False
            self.Nparams = 1
            self.param_names = ['Zmet']
            self.param_descr = ['Metallicity (mass fraction, where solar = 0.020)']
            self.param_names_fncy = [r'$Z$']
            self.param_bounds = np.array([1e-5, 0.040]).reshape(1,2)

        #lnu_model = np.array(burst_dict['lnu']) # Lnu[wave, time] (rest frame)
        # wave_lines = burst_dict['wlines'] # Wavelength of lines
        # l_lines = burst_dict['l_lines'] # Integrated luminosity of lines


        wave_model_obs = wave_model * (1 + self.redshift)
        nu_model_obs = nu_model / (1 + self.redshift)

        lnu_obs = lnu_model * (1 + self.redshift)

        if (self.step):

            Nbins = len(self.age) - 1
            dt_bins = np.array(self.age[1:]) - np.array(self.age[:-1])
            if np.any(dt_bins < 10**6.05):
                raise ValueError('The minimum age bin width is 10**6.05 years; this is set by the time resolution of the source models.')
            self.Nages = Nbins

            q0_age = np.zeros((Nbins,len(self.Zmet)), dtype='double') # Ionizing photons per bin
            if (nebular_effects):
                lnu_age = np.zeros((Nbins, len(self.Zmet), len(self.logU), len(wave_model)), dtype='double') # Lnu(wave) per bin and logU
                llines_age = np.zeros((Nbins, len(self.Zmet), len(self.logU), len(self.line_names)), dtype='double')
            else:
                lnu_age = np.zeros((Nbins, len(self.Zmet), len(wave_model)), dtype='double') # Lnu(wave) per bin
                llines_age = np.zeros((Nbins, len(self.Zmet), len(self.line_names)))
            lbol_age = np.zeros((Nbins,len(self.Zmet)), dtype='double') # Bolometric luminosity in bin
            mstar_age = np.zeros((Nbins,len(self.Zmet)), dtype='double') # Mass in bin

            #dt = 5.e5 # Time resolution in years
            #n_substeps = 100 # number of time divisions -- faster to do it this way, seemingly no loss of accuracy
            #t0 = time_module.time()
            for i in np.arange(Nbins):

                #if (i == Nbins - 1): dt = 1e6

                ti = self.age[i]
                tf = self.age[i + 1]
                # bin_width = tf - ti
                # #n_substeps = (bin_width // dt) # Number of timesteps in bin
                # dt = bin_width / n_substeps
                # # The integral/convolution is Lnu_j = int(Lnu(tf - t) dt, 0, tf - ti);
                # # The observed stellar population of a bin formed at the start of the bin and aged tf - ti years.
                # time_substeps = dt * np.arange(n_substeps + 1) # Our linearly spaced time subgrid.
                # integrate_here = (time_substeps >= 0) & (time_substeps <= bin_width)
                # q0_age[i] = trapz(np.interp(tf - time_substeps, time, q0)[integrate_here], time_substeps[integrate_here])
                # lbol_age[i] = trapz(np.interp(tf - time_substeps, time, lbol)[integrate_here], time_substeps[integrate_here])
                # mstar_age[i] = trapz(np.interp(tf - time_substeps, time, mstar)[integrate_here], time_substeps[integrate_here])
                #
                # # Vectorize later
                # lnu_finterp = interp1d(time, lnu_obs, axis=0)
                # lnu_age[i,...] = trapz(lnu_finterp(tf - time_substeps)[integrate_here, ...], time_substeps[integrate_here], axis=0)
                # # if nebular_effects:
                # #
                # # else:
                # #     for j in np.arange(len(nu_model_obs)):
                # #         lnu_age[i,j] = trapz(np.interp(tf - time_substeps, time, lnu_obs[j,:])[integrate_here], time_substeps[integrate_here])
                #
                # #lnu_age[i,...] = trapz(np.interp(tf - time_substeps, time, lnu_obs[j,:])[integrate_here], time_substeps[integrate_here])


                # Alternate scheme: how many whole source bins can we fit into [ti, tf]? And then we just interpolate
                # the first and last bins only. this avoids the problems presented by having dt0 = 10**6.05... So we have a cumulative
                # delta t array (or just the upper edge of the age bin, same thing) and we look for the largest index such that
                # tbin_hi < (tf - ti), then we truncate the partial bin (if there is one, though there basically always
                # will be because the bins are *centered* on round numbers and have 0.05 dex widths)
                # fullbin_mask = time_hi <= (tf) & (time_lo > ti)
                # partial_bin = (np.count_nonzero(time_hi == (tf - ti))) == 0
                #
                # q0_age[i] = np.sum(q0[fullbin_mask] * deltat[fullbin_mask])
                # lbol_age[i] = np.sum(lbol[fullbin_mask] * deltat[fullbin_mask])
                # mstar_age[i] = np.sum(mstar[fullbin_mask] * deltat[fullbin_mask])
                # lnu_age[i,...] = np.squeeze(np.sum(np.atleast_3d(lnu_obs[fullbin_mask,...]) * deltat[fullbin_mask,None,None], axis=0))
                #
                # if partial_bin:
                #     # Diff works on boolean arrays; here we get index of last full bin + 1
                #     indexof = np.argmax(np.diff(fullbin_mask)) + 1
                #     deltat_partial = (tf - ti) - time_lo[indexof] # is a scalar
                #
                #     q0_age[i] += q0[indexof] * deltat_partial
                #     lbol_age[i] += lbol[indexof] * deltat_partial
                #     mstar_age[i] += mstar[indexof] * deltat_partial
                #     lnu_age[i,...] += lnu_obs[indexof,...] * deltat_partial

                # tf and ti are the limits of integration, time_lo and time_hi are the limits of the source bins.
                fullbins = (time_hi < tf) & (time_lo >= ti)
                partialbinlo = ((time_lo < ti) & (time_hi >= ti))
                partialbinhi = ((time_lo < tf) & (time_hi >= tf))
                q0_age[i,:] = np.sum(q0[fullbins,:] * deltat[fullbins,None], axis=0)
                lbol_age[i,:] = np.sum(lbol[fullbins,:] * deltat[fullbins,None], axis=0)
                mstar_age[i,:] = np.sum(mstar[fullbins,:] * deltat[fullbins,None], axis=0)
                # lnu_age[i,...] = np.squeeze(np.sum(np.atleast_3d(lnu_obs[fullbins,...]) * deltat[fullbins,None,None], axis=0))
                if nebular_effects:
                    llines_age[i,...] = np.sum(llines[fullbins,...] * deltat[fullbins,None,None,None], axis=0)
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
                        llines_age[i,...] += np.squeeze(llines[partialbinlo,...] * deltat_partial)
                if np.any(partialbinhi):
                    deltat_partial = tf - time_lo[partialbinhi]
                    q0_age[i,:] += np.squeeze(q0[partialbinhi,:] * deltat_partial)
                    lbol_age[i,:] += np.squeeze(lbol[partialbinhi,:] * deltat_partial)
                    mstar_age[i,:] += np.squeeze(mstar[partialbinhi,:] * deltat_partial)
                    lnu_age[i,...] += np.squeeze(lnu_obs[partialbinhi,...] * deltat_partial)
                    if nebular_effects:
                        llines_age[i,...] += np.squeeze(llines[partialbinhi,...] * deltat_partial)

        else:

            Nages = len(self.age)
            self.Nages = Nages

            q0_age = np.zeros((self.Nages, len(self.Zmet)))
            lbol_age = np.zeros((self.Nages, len(self.Zmet)))
            mstar_age = np.zeros((self.Nages, len(self.Zmet)))
            # mstar_rem_age = np.zeros((self.Nages, len(self.Zmet)))

            if (nebular_effects):
                lnu_age = np.zeros((self.Nages, len(self.Zmet), len(self.logU), len(wave_model)), dtype='double') # Lnu(wave) per bin and logU
                llines_age = np.zeros((self.Nages, len(self.Zmet), len(self.logU), len(self.line_names)), dtype='double')

            else:
                lnu_age = np.zeros((self.Nages, len(self.Zmet), len(wave_model)), dtype='double') # Lnu(wave) per bin
                llines_age = np.zeros((self.Nages, len(self.Zmet), len(self.line_names)))


            #t0 = time_module.time()

            for j in np.arange(len(self.Zmet)):
                q0_age[:,j] = np.interp(self.age, time, q0[:,j])
                lbol_age[:,j] = np.interp(self.age, time, lbol[:,j])
                mstar_age[:,j] = np.interp(self.age, time, mstar[:,j])

            lnu_finterp = interp1d(time, lnu_obs, axis=0)
            lnu_age = lnu_finterp(self.age)
            # for j in np.arange(len(nu_model_obs)):
            #     lnu_age[:,j] = np.interp(age, time, lnu_obs[j,:])

            if (nebular_effects):
                llines_finterp = interp1d(time, llines, axis=0)
                llines_age = llines_finterp(self.age)

        #t1 = time_module.time()

        self.mstar = mstar_age
        self.Lbol = lbol_age
        self.q0 = q0_age
        self.line_lum = llines_age

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
            assert (params is not None), 'BPASS models with the Cloudy component enabled require logU to be specified.'
            assert (params.shape[0] == Nmodels), 'First dimension of logU array must match first dimension of SFH.'
            ob_mask = self._check_bounds(params)
            if np.any(ob_mask):
                raise ValueError('%d logU value(s) are out of bounds [-4,-1]' % (np.count_nonzero(ob_mask)))

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
            #print('Lnu_obs gridded shape: ', self.Lnu_obs.shape, len(self.Lnu_obs.shape))
            Zmet = params[:,0]
            logU = params[:,1]

            Lnu_obs = 10**interpn((self.Zmet, self.logU),
                                   np.log10(np.transpose(self.Lnu_obs, axes=[1,2,0,3])),
                                   params,
                                   method='linear')

            #finterp = interp1d(self.logU, self.Lnu_obs, axis=1)
            # Lnu_obs = finterp(params)
            # # params comes in as (Nmodels, 1) so Lnu_obs will have shape ()
            # if len(Lnu_obs.shape) > 3:
            #     Lnu_obs = np.squeeze(Lnu_obs, axis=2)
            #Lnu_obs = finterp(params.flatten())
            #Lnu_obs = np.squeeze(finterp(params)) # (Nages, Nmodels, Nwave)
            #print('Lnu_obs shape after interpolation: ', Lnu_obs.shape, len(Lnu_obs.shape))
            #Lnu_obs = np.swapaxes(Lnu_obs, 0, 1) # (Nmodels, Nages, Nwave)
        else:
            # We need only interpolate in the metallicity dimension.
            finterp = interp1d(self.Zmet, np.log10(self.Lnu_obs), axis=1)
            Lnu_obs = 10**finterp(params.flatten())
            #print('Shape after interpolation: ', Lnu_obs.shape)
            Lnu_obs = np.swapaxes(Lnu_obs, 0, 1)

            #Lnu_obs = self.Lnu_obs # (Nmodels, Nages, Nwave)

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
                #print(sfh_param.shape)
                #print(params.shape)
                #print('Lnu_obs shape:', Lnu_obs.shape, len(Lnu_obs.shape))
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
            assert (params is not None), 'BPASS models with the Cloudy component enabled require logU to be specified.'
            assert (params.shape[0] == Nmodels), 'First dimension of logU array must match first dimension of SFH.'

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
            Values for logU, if the model includes a nebular component.
        stepwise : bool
            If true, the spectrum is returned as a function of stellar age.

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
        assert (params.shape[0] == Nmodels), 'First dimension of logU array must match first dimension of SFH.'

        #print(self.line_lum)
        L_lines = 10**interpn((self.Zmet, self.logU),
                             np.log10(np.transpose(self.line_lum, axes=[1,2,0,3])),
                             params,
                             method='linear')
        #print(L_lines)
        # Send the originally 0 luminosities made NaN by log interpolation back to 0.
        L_lines[np.isnan(L_lines)] = 0

        # finterp = interp1d(self.logU, self.line_lum, axis=1)
        # L_lines = finterp(params.flatten()) # (Nages, Nmodels, Nlines)
        #L_lines = np.swapaxes(L_lines, 0, 1) # (Nmodels, Nages, Nlines)

        if stepwise:
            ages_Lmod_lines = sfh.multiply(sfh_param, L_lines)

            return ages_Lmod_lines

        else:
            if (self.step):
                Lmod_lines = sfh.sum(sfh_param, L_lines)
            else:
                Lmod_lines = sfh.integrate(sfh_param, L_lines)

            return Lmod_lines


class BPASSModelA24(BaseEmissionModel):
    '''Stellar emission models generated using BPASS, including
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
    nebula_old : bool
        If ``True``, the spectra will include nebular extinction and emission at ages older than 50 Myr, modeling
        the nebular contributions of the hot, stripped cores of evolved massive stars. While the the conditions we assume
        for the nebula are most appropriate for massive H II regions, Byler+(2017) found that nebular emission from
        post-AGB stars is not super sensitive to the geometry/density of the nebula, but it may be worth fitting your
        galaxies with and without this component if you're concerned.
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

    model_name = 'BPASS-Stellar-A24'
    model_type = 'Stellar-Emission'
    gridded = False

    def _construct_model(self, age=None, lognH=2.0, step=True,
                         wave_grid=None, cosmology=None, binaries=True,
                         nebular_effects=True, nebula_old=True, dust_grains=False):
        '''
            Load the appropriate models from the BPASS h5 files and either integrate
            them in bins (if ``step==True``) or interpolate them to an age grid otherwise.
        '''
        self.path_to_models = self.path_to_models + 'BPASS_Cloudy/imf_chab300/' + 'BPASS_imf_chab300_fullgrid_ng.h5'
        f = h5py.File(self.path_to_models)

        self.Zmet = f['Zstars'][:]

        if (nebular_effects) and (not binaries):
            raise ValueError('Binaries are required to use the Cloudy nebular emission grids.')

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
        self.line_names = f['lines/names'][:]
        #Nlines = len(linenames)

        nhmask = np.log10(self.nH) == lognH

        if (nebular_effects):

            # Leave the nebula in place around old stars?
            # This models nebular emission driven by e.g. post-AGB stars
            # which are hot enough to produce significant ionizing flux for
            # a large enough stellar mass. However,
            if nebula_old:
                # Squeeze removes the singleton dimension where density is indexed
                lnu_model = np.squeeze(f['spec/neb'][:,:,:,nhmask,:])
            else:
                # This cut excludes stellar populations older than about 50 Myr from the nebular
                # modeling.
                lnu_model = np.zeros((len(time), len(self.Zmet), len(self.logU), len(wave_model)))
                lnu_model[:16,:,:,:] = f['spec/neb'][:16,:,:,nhmask,:]
                lnu_model[16:,:,:,:] = (f['spec/noneb'][16:,:])[:,:,None,:]
            # lineratios = f['lines/ratios'][:,:,:,:]
            linelum = np.squeeze(f['lines/lum'][:,:,:,nhmask,:])
            mstar = f['mstar'][:,:]
            mstar_rem = f['mstar_remnants'][:,:]
            q0 = 10**f['logq0'][:,:]
            lbol = f['Lbol'][:,:]

            # lineratios[np.isnan(lineratios)] = 0.0

            self.nebular = True
            self.Nparams = 2
            self.param_names = ['Zmet', 'logU']
            self.param_descr = ['Metallicity (mass fraction, where solar = 0.020 ~ 10**[-1.7])',
                                'log10 of the ionization parameter']
            self.param_names_fncy = [r'$Z$', r'$\log \mathcal{U}$']
            self.param_bounds = np.array([[np.min(self.Zmet), np.max(self.Zmet)],
                                          [-4, -1.5]])
        else:

            if binaries:
                lnu_model = f['spec/noneb'][:,:,:]
                mstar = f['mstar'][:]
                mstar_rem = f['mstar_remnants'][:,:]
                q0 = 10**f['logq0'][:,:]
                lbol = f['Lbol'][:,:]
            else:
                lnu_model = f['singlestars/spec'][:,:,:]
                mstar = f['singlestars/mstar'][:]
                mstar_rem = np.zeros((len(time), len(self.Zmet)))
                q0 = 10**f['singlestars/logq0'][:,:]
                lbol = f['singlestars/Lbol'][:,:]

            # lineratios = np.zeros((len(time), len(self.Zmet), len(self.line_names)))

            self.nebular = False
            self.Nparams = 1
            self.param_names = ['Zmet']
            self.param_descr = ['Metallicity (mass fraction, where solar = 0.020 ~ 10**[-1.7])']
            self.param_names_fncy = [r'$Z$']
            self.param_bounds = np.array([np.min(self.Zmet), np.max(self.Zmet)]).reshape(1,2)

        wave_model_obs = wave_model * (1 + self.redshift)
        nu_model_obs = nu_model / (1 + self.redshift)

        lnu_obs = lnu_model * (1 + self.redshift)

        if (self.step):

            Nbins = len(self.age) - 1
            dt_bins = np.array(self.age[1:]) - np.array(self.age[:-1])
            if np.any(dt_bins < 10**6.05):
                raise ValueError('The minimum age bin width is 10**6.05 years; this is set by the time resolution of the source models.')
            self.Nages = Nbins

            q0_age = np.zeros((Nbins,len(self.Zmet)), dtype='double') # Ionizing photons per bin
            if (nebular_effects):
                lnu_age = np.zeros((Nbins, len(self.Zmet), len(self.logU), len(wave_model)), dtype='double') # Lnu(wave) per bin and logU
                #lineratios_age = np.zeros((Nbins, len(self.Zmet), len(self.logU), len(self.line_names)), dtype='double')
                linelum_age = np.zeros((Nbins, len(self.Zmet), len(self.logU), len(self.line_names)), dtype='double')
            else:
                lnu_age = np.zeros((Nbins, len(self.Zmet), len(wave_model)), dtype='double') # Lnu(wave) per bin
                #lineratios_age = np.zeros((Nbins, len(self.Zmet), len(self.line_names)))
                linelum_age = np.zeros((Nbins, len(self.Zmet), len(self.line_names)))
            lbol_age = np.zeros((Nbins,len(self.Zmet)), dtype='double') # Bolometric luminosity in bin
            mstar_age = np.zeros((Nbins,len(self.Zmet)), dtype='double') # Mass in bin
            mstar_rem_age = np.zeros((Nbins,len(self.Zmet)), dtype='double')

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
                mstar_rem_age[i,:] = np.sum(mstar_rem[fullbins,:] * deltat[fullbins,None], axis=0)
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
                    mstar_rem_age[i,:] += np.squeeze(mstar_rem[partialbinlo,:] * deltat_partial)
                    lnu_age[i,...] += np.squeeze(lnu_obs[partialbinlo,...] * deltat_partial)
                    if nebular_effects:
                        linelum_age[i,...] += np.squeeze(linelum[partialbinlo,...] * deltat_partial)
                if np.any(partialbinhi):
                    deltat_partial = tf - time_lo[partialbinhi]
                    q0_age[i,:] += np.squeeze(q0[partialbinhi,:] * deltat_partial)
                    lbol_age[i,:] += np.squeeze(lbol[partialbinhi,:] * deltat_partial)
                    mstar_age[i,:] += np.squeeze(mstar[partialbinhi,:] * deltat_partial)
                    mstar_rem_age[i,:] += np.squeeze(mstar_rem[partialbinhi,:] * deltat_partial)
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
            mstar_rem_age = np.zeros((self.Nages, len(self.Zmet)))


            if (nebular_effects):
                lnu_age = np.zeros((self.Nages, len(self.Zmet), len(self.logU), len(wave_model)), dtype='double') # Lnu(wave) per bin and logU
                linelum_age = np.zeros((self.Nages, len(self.Zmet), len(self.logU), len(self.line_names)), dtype='double')

            else:
                lnu_age = np.zeros((self.Nages, len(self.Zmet), len(wave_model)), dtype='double') # Lnu(wave) per bin
                linelum_age = np.zeros((self.Nages, len(self.Zmet), len(self.line_names)))

            for j in np.arange(len(self.Zmet)):
                q0_age[:,j] = np.interp(self.age, time, q0[:,j])
                lbol_age[:,j] = np.interp(self.age, time, lbol[:,j])
                mstar_age[:,j] = np.interp(self.age, time, mstar[:,j])
                mstar_rem_age[:,j] = np.interp(self.age, time, mstar_rem[:,j])

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
            assert (params is not None), 'BPASS models with the Cloudy component enabled require logU to be specified.'
            assert (params.shape[0] == Nmodels), 'First dimension of logU array must match first dimension of SFH.'
            ob_mask = self._check_bounds(params)
            if np.any(ob_mask):
                raise ValueError('%d logU value(s) are out of bounds [-4,-1]' % (np.count_nonzero(ob_mask)))

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

            Lnu_obs = 10**interpn((self.Zmet, self.logU),
                                   np.log10(np.transpose(self.Lnu_obs, axes=[1,2,0,3])),
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
            assert (params is not None), 'BPASS models with the Cloudy component enabled require logU to be specified.'
            assert (params.shape[0] == Nmodels), 'First dimension of logU array must match first dimension of SFH.'

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
            Values for logU, if the model includes a nebular component.
        stepwise : bool
            If true, the spectrum is returned as a function of stellar age.

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
        assert (params.shape[0] == Nmodels), 'First dimension of logU array must match first dimension of SFH.'

        L_lines = 10**interpn((self.Zmet, self.logU),
                               np.log10(np.transpose(self.line_lum, axes=[1,2,0,3])),
                               params,
                               method='linear')

        # Send the originally 0 luminosities made NaN by log interpolation back to 0.
        L_lines[np.isnan(L_lines)] = 0

        if stepwise:
            ages_Lmod_lines = sfh.multiply(sfh_param, L_lines)

            return ages_Lmod_lines

        else:
            if (self.step):
                Lmod_lines = sfh.sum(sfh_param, L_lines)
            else:
                Lmod_lines = sfh.integrate(sfh_param, L_lines)

            return Lmod_lines

class BPASSBurstA24(BPASSModelA24):
    '''
    SFH-free model representing a single instantaneous burst
    of star formation with a given mass and age.
    '''

    def __init__(self, filter_labels, redshift, wave_grid=None, age=None, lognH=2.0, cosmology=None):

        if cosmology is None:
            from astropy.cosmology import FlatLambdaCDM
            cosmology = FlatLambdaCDM(H0=70, Om0=0.3)

        univ_age = cosmology.age(redshift).value * 1e9

        # "Erik googled how super() works after 15 years"
        super().__init__(filter_labels, redshift, step=False, wave_grid=wave_grid, age=age, lognH=lognH, cosmology=cosmology)

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

        finterp_lnu = interp1d(self.age, np.log10(self.Lnu_obs), axis=0)

        # The axes go (age, Z, logU, wave)
        lnu_unattenuated = 10**interpn((self.age, self.Zmet, self.logU),
                                 np.log10(self.Lnu_obs),
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


        if (self.step):
            assert (sfh.type == 'piecewise'), 'Binned stellar populations require a piecewise-defined SFH.'

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

        if (len(params.shape) == 1):
            params = params.reshape(1, -1)

        Nmodels = params.shape[0]
        Mburst = params[:,0]

        L_lines = 10**interpn((self.age, self.Zmet, self.logU),
                               np.log10(self.line_lum),
                               params[:,1:],
                               method='linear')

        # Send the originally 0 luminosities made NaN by log interpolation back to 0.
        L_lines[np.isnan(L_lines)] = 0
        L_lines *= Mburst[:,None]

        return L_lines
