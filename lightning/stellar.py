'''
    stellar.py

    Stellar population modeling.
'''

import sys
import os
import numpy as np
import time as time_module
from pathlib import Path

#from astropy.io import fits
import astropy.units as u
import astropy.constants as const
#from astropy.table import Table
#import pysynphot as SP

from scipy.interpolate import interp1d
from scipy.integrate import trapz
from scipy.io import readsav
#from scipy import stats
#import scipy.optimize as opt

__all__ = ['StellarModel']

#################################
# Build stellar pop'ns
# from PEGASE
#################################
class StellarModel:
    '''
        Stellar emission models (as of right now) generated using Pégase.

        These models are either:
            - A single burst of star formation at 1 solar mass yr-1, evaluated
              on a grid of specified ages
            - A binned stellar population, representing a constant epoch of star
              formation in a specified set of stellar age bins. These models are
              integrated from the above.

        Nebular extinction + continuum are included by default, lines are optional,
        added by hand.
    '''

    #param_names = ['sfh']
    #param_descr = ['Star formation history: as of right now this is an array giving SFR in solar masses yr-1 at each stellar age.']

    def __init__(self, filter_labels, redshift, age,
                 step=True, Z_met=0.020, add_lines=True, wave_grid=None,
                 path_to_filters=None, path_to_models=None):

        if (path_to_models is None):
            self.path_to_models = str(Path(__file__).parent.resolve()) + '/models/04-single_burst/Kroupa01/'
        else:
            self.path_to_models = path_to_models
            if(self.path_to_models[-1] != '/'): self.path_to_models = self.path_to_models + '/'


        if (path_to_filters is None):
            self.path_to_filters = str(Path(__file__).parent.resolve()) + '/filters/'
        else:
            self.path_to_filters = path_to_filters
            if(self.path_to_filters[-1] != '/'): self.path_to_filters = self.path_to_filters + '/'

        self.path_to_models = self.path_to_models + 'Kroupa01_Z%5.3f_nebular_spec.idl' % (Z_met)
        self.age = age
        self.redshift = redshift
        self.filter_labels = filter_labels
        self.step = step
        self.metallicity = Z_met

        burst_dict = readsav(self.path_to_models)

        # These are views into the original dict and so
        # are read-only.
        wave_model = burst_dict['wave'] # Wavelength grid, rest frame
        nu_model = burst_dict['nu'] # Freq grid, rest frame
        time = burst_dict['time'] # Time grid
        # lnu_model is more convenient if it's assignable, so we'll make
        # a copy.
        lnu_model = np.array(burst_dict['lnu']) # Lnu[wave, time] (rest frame)
        wave_lines = burst_dict['wlines'] # Wavelength of lines
        l_lines = burst_dict['l_lines'] # Integrated luminosity of lines

        mstar = burst_dict['mstars'] # Stellar mass
        q0 = burst_dict['nlyc'] # Number of lyman continuum photons
        lbol = burst_dict['lbol'] # Bolometric luminosity

        wave_model_obs = wave_model * (1 + self.redshift)
        nu_model_obs = nu_model / (1 + self.redshift)

        # Handle emission lines: Add a narrow gaussian
        # to the model at the location of each line
        dv = 50 * u.km / u.s
        z = (dv / const.c.to(u.km / u.s)).value

        if(add_lines):
            # For each timestep...
            for j,t in enumerate(time):
                lnu_lines = np.zeros(len(burst_dict['wave']))
                # For each line...
                for i,wave in enumerate(wave_lines):
                    lnu_line = np.exp(-1.0 * (((wave_model / wave) - 1) / z)**2)
                    lnu_line = lnu_line / np.abs(trapz(lnu_line, nu_model))
                    lnu_lines = lnu_lines + l_lines[i,j] * lnu_line

                lnu_model[:,j] = lnu_model[:,j] + lnu_lines

        lnu_obs = lnu_model * (1 + self.redshift)

        if (self.step):

            Nbins = len(age) - 1
            self.Nages = Nbins

            q0_age = np.zeros(Nbins, dtype='double') # Ionizing photons per bin
            lnu_age = np.zeros((Nbins, len(wave_model)), dtype='double') # Lnu(wave) per bin
            lbol_age = np.zeros(Nbins, dtype='double') # Bolometric luminosity in bin
            mstar_age = np.zeros(Nbins, dtype='double') # Mass in bin

            #dt = 5.e5 # Time resolution in years
            n_substeps = 100 # number of time divisions -- faster to do it this way, seemingly no loss of accuracy
            t0 = time_module.time()
            for i in np.arange(Nbins):

                if (i == Nbins - 1): dt = 1e6

                ti = age[i]
                tf = age[i + 1]
                bin_width = tf - ti
                #n_substeps = (bin_width // dt) # Number of timesteps in bin
                dt = bin_width / n_substeps
                time_substeps = dt * np.arange(n_substeps + 1) # "Time from the onset of SF, progressing linearly"
                integrate_here = (time_substeps >= 0) & (time_substeps <= bin_width)
                q0_age[i] = trapz(np.interp(tf - time_substeps, time, q0)[integrate_here], time_substeps[integrate_here])
                lbol_age[i] = trapz(np.interp(tf - time_substeps, time, lbol)[integrate_here], time_substeps[integrate_here])
                mstar_age[i] = trapz(np.interp(tf - time_substeps, time, mstar)[integrate_here], time_substeps[integrate_here])

                # Vectorize later
                for j in np.arange(len(nu_model_obs)):
                    lnu_age[i,j] = trapz(np.interp(tf - time_substeps, time, lnu_obs[j,:])[integrate_here], time_substeps[integrate_here])

        else:

            Nages = len(age)
            self.Nages = Nages

            lnu_age = np.zeros((Nages, len(wave_model)), dtype='double') # Lnu(wave)

            t0 = time_module.time()

            q0_age = np.interp(age, time, q0)
            lbol_age = np.interp(age, time, lbol)
            mstar_age = np.interp(age, time, mstar)

            # Vectorize later
            for j in np.arange(len(nu_model_obs)):
                lnu_age[:,j] = np.interp(age, time, lnu_obs[j,:])

        t1 = time_module.time()

        self.mstar = mstar_age
        self.Lbol = lbol_age
        self.q0 = q0_age

        c_um = const.c.to(u.micron / u.s).value

        if (wave_grid is not None):
            finterp = interp1d(wave_model, lnu_age, bounds_error=False, fill_value=0.0, axis=1)
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

        # Get filters
        self.filters = dict()
        for name in self.filter_labels:
            self.filters[name] = np.zeros(len(self.wave_grid_rest), dtype='float')

        self._get_filters()
        self.Nfilters = len(self.filters)

        self.wave_obs = np.zeros(len(self.filter_labels), dtype='float')
        self._get_wave_obs()
        self.nu_obs = c_um / self.wave_obs


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


    def get_model_lnu_hires(self, sfh, sfh_param, exptau=None, exptau_youngest=None, stepwise=False):

        # sfh_shape = sfh.shape # expecting ndarray(Nmodels, n_steps)
        if (len(sfh_param.shape) == 1):
            sfh_param = sfh_param.reshape(1, -1)

        if (self.step):
            assert (sfh.type == 'piecewise'), 'Binned stellar populations require a piecewise-defined SFH.'

        Nmodels = sfh_param.shape[0]
        #n_steps = sfh_param_shape[1]

        # Explicit dependence of the attenuation on stage is not currently fully implemented, but would be easy to
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

        # It is sometimes useful to have the spectra evaluated at each stellar age
        if stepwise:

            ages_lnu_unattenuated = sfh.multiply(sfh_param, self.Lnu_obs)
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

                lnu_unattenuated = sfh.sum(sfh_param, self.Lnu_obs)
                # lnu_attenuated = lnu_unattenuated.copy()
                # lnu_attenuated = lnu_unattenuated * exptau
                # L_TIR = np.abs(trapz(lnu_unattenuated - lnu_attenuated, self.nu_grid_obs, axis=1))

            else:
                # lnu_unattenuated = trapz(ages_lnu_unattenuated, self.age, axis=1)
                # lnu_attenuated = trapz(ages_lnu_attenuated, self.age, axis=1)
                # L_TIR = trapz(ages_L_TIR, self.age, axis=1)

                lnu_unattenuated = sfh.integrate(sfh_param, self.Lnu_obs)

            lnu_attenuated = lnu_unattenuated.copy()
            lnu_attenuated = lnu_unattenuated * exptau
            L_TIR = np.abs(trapz(lnu_unattenuated - lnu_attenuated, self.nu_grid_obs, axis=1))

            if (Nmodels == 1):
                lnu_unattenuated = lnu_unattenuated.flatten()
                lnu_attenuated = lnu_attenuated.flatten()
                L_TIR = L_TIR.flatten()

            return lnu_attenuated, lnu_unattenuated, L_TIR


    def get_model_lnu(self, sfh, sfh_param, exptau=None, exptau_youngest=None, stepwise=False):

        # sfh_shape = sfh.shape # expecting ndarray(Nmodels, Nages)
        #
        # if (len(sfh_shape) == 1):
        #     sfh = sfh.reshape(1, sfh.size)
        #     sfh_shape = sfh.shape

        if (len(sfh_param.shape) == 1):
            sfh_param = sfh_param.reshape(1, -1)


        if (self.step):
            assert (sfh.type == 'piecewise'), 'Binned stellar populations require a piecewise-defined SFH.'

        Nmodels = sfh_param.shape[0]
        #Nages = sfh_shape[1]

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

            # Integrate steps_lnu_unattenuated - steps_lnu_attenuated to get the dust luminosity per bin
            # Comes out negative since self.nu_grid_obs is monotonically decreasing.
            #ages_L_TIR = np.abs(trapz(ages_lnu_unattenuated - ages_lnu_attenuated, self.nu_grid_obs, axis=2))

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
            # lmod_unattenuated = np.sum(ages_lmod_unattenuated, axis=1)
            # lmod_attenuated = np.sum(ages_lmod_attenuated, axis=1)
            # L_TIR = np.sum(ages_L_TIR, axis=1)

            lnu_attenuated, lnu_unattenuated, L_TIR = self.get_model_lnu_hires(sfh, sfh_param, exptau=exptau, exptau_youngest=exptau_youngest, stepwise=True)

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


# def steps_stellar(z_gal, age_bin_edges, Z_met=0.020, add_lines=False, wave_grid=None, path='/Users/Erik/Documents/Research/lightning/plightning/models/04-single_burst/Kroupa01/'):
#     '''
#         Load the stellar
#         populations, and integrate the bursts
#         from PEGASE into the binned stellar
#         population models we want.
#
#         The lightning models are normalized to
#         1 solar mass yr-1. They're given in units
#         of Lnu (observed frame frequency).
#     '''
#
#     # Make sure that given age bins are allowed
#     # given the age of the Universe at z_gal
#
#     # Bad to hardcode this, fix it later.
#     #pegase_fname = '/Users/Erik/Documents/Research/lightning/stable/models/04-single_burst/Kroupa01/Kroupa01_Z0.013_nebular_spec.idl'
#
#     pegase_fname = path + 'Kroupa01_Z%5.3f_nebular_spec.idl' % (Z_met)
#
#     burst_dict = readsav(pegase_fname)
#
#     # These are views into the original dict and so
#     # are read-only.
#     wave_model = burst_dict['wave'] # Wavelength grid, rest frame
#     nu_model = burst_dict['nu'] # Freq grid, rest frame
#     time = burst_dict['time'] # Time grid
#     # lnu_model is more convenient if it's assignable, so we'll make
#     # a copy.
#     lnu_model = np.array(burst_dict['lnu']) # Lnu[wave, time] (rest frame)
#     wave_lines = burst_dict['wlines'] # Wavelength of lines
#     l_lines = burst_dict['l_lines'] # Integrated luminosity of lines
#
#     mstar = burst_dict['mstars']
#     q0 = burst_dict['nlyc']
#     lbol = burst_dict['lbol']
#
#     wave_model_obs = wave_model * (1 + z_gal)
#     nu_model_obs = nu_model / (1 + z_gal)
#
#     # Handle emission lines:
#     # Add a narrow gaussian
#     # to the model at the location
#     # of each line
#     dv = 50 * u.km / u.s
#     z = (dv / const.c.to(u.km / u.s)).value
#
#     if(add_lines):
#         # For each timestep...
#         for j,t in enumerate(time):
#             lnu_lines = np.zeros(len(burst_dict['wave']))
#             # For each line...
#             for i,wave in enumerate(wave_lines):
#                 lnu_line = np.exp(-1.0 * (((wave_model / wave) - 1) / z)**2)
#                 lnu_line = lnu_line / np.abs(trapz(lnu_line, nu_model))
#                 lnu_lines = lnu_lines + l_lines[i,j] * lnu_line
#
#             lnu_model[:,j] = lnu_model[:,j] + lnu_lines
#
#     lnu_obs = lnu_model * (1 + z_gal)
#
#     Nbins = len(age_bin_edges) - 1
#
#     q0_steps = np.zeros(Nbins, dtype='double') # Ionizing photons per bin
#     lnu_steps = np.zeros((Nbins, len(wave_model)), dtype='double') # Lnu(wave) per bin
#     lbol_steps = np.zeros(Nbins, dtype='double') # Bolometric luminosity in bin
#     mstar_steps = np.zeros(Nbins, dtype='double') # Mass in bin
#
#     #dt = 5.e5 # Time resolution in years
#     n_substeps = 100 # number of time divisions
#     t0 = time_module.time()
#     for i in np.arange(Nbins):
#
#         if (i == Nbins - 1): dt = 1e6
#
#         ti = age_bin_edges[i]
#         tf = age_bin_edges[i + 1]
#         bin_width = tf - ti
#         #n_substeps = (bin_width // dt) # Number of timesteps in bin
#         dt = bin_width / n_substeps
#         time_substeps = dt * np.arange(n_substeps + 1) # "Time from the onset of SF, progressing linearly"
#         integrate_here = (time_substeps >= 0) & (time_substeps <= bin_width)
#         q0_steps[i] = trapz(np.interp(tf - time_substeps, time, q0)[integrate_here], time_substeps[integrate_here])
#         lbol_steps[i] = trapz(np.interp(tf - time_substeps, time, lbol)[integrate_here], time_substeps[integrate_here])
#         mstar_steps[i] = trapz(np.interp(tf - time_substeps, time, mstar)[integrate_here], time_substeps[integrate_here])
#
#         # Vectorize later
#         for j in np.arange(len(nu_model_obs)):
#             lnu_steps[i,j] = trapz(np.interp(tf - time_substeps, time, lnu_obs[j,:])[integrate_here], time_substeps[integrate_here])
#
#     t1 = time_module.time()
#
#     if (wave_grid is not None):
#         finterp = interp1d(wave_model, lnu_steps, bounds_error=False, fill_value=0.0, axis=1)
#         lnu_steps_interp = finterp(wave_grid)
#         lnu_steps_interp[lnu_steps_interp < 0.0] = 0.0
#         nu_grid = const.c.to(u.micron / u.s) / wave_grid
#         out_dict = {'wave_rest': wave_grid,
#                     'wave_obs': wave_grid * (1 + z_gal),
#                     'nu_rest': nu_grid,
#                     'nu_obs': nu_grid * (1 + z_gal),
#                     'lnu_obs': lnu_steps_interp,
#                     'mstar': mstar_steps,
#                     'lbol': lbol_steps,
#                     'q0': q0_steps,
#                     'age_bins': age_bin_edges,
#                     'redshift': z_gal}
#     else:
#         out_dict = {'wave_rest': wave_model,
#                     'wave_obs': wave_model_obs,
#                     'nu_rest': nu_model,
#                     'nu_obs': nu_model_obs,
#                     'lnu_obs': lnu_steps,
#                     'mstar': mstar_steps,
#                     'lbol': lbol_steps,
#                     'q0': q0_steps,
#                     'age_bins': age_bin_edges,
#                     'redshift': z_gal}
#
#
#     #print('steps_stellar: %.2f seconds elapsed' % (t1 - t0))
#
#     return out_dict
