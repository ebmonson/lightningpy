import warnings
import numbers
from pathlib import Path

import numpy as np
from scipy.integrate import trapz
from scipy.interpolate import interp1d

import astropy.constants as const
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM
from astropy.utils.exceptions import AstropyUserWarning

from ..base import BaseEmissionModel
from ..get_filters import get_filters

class XrayEmissionModel(BaseEmissionModel):
    '''Base class for X-ray emission models.
    '''

    Nparams = 0
    model_name = 'None'
    model_type = 'None'
    gridded = False
    param_names = ['None']
    param_descr = ['None']
    param_names_fncy = [r'None']
    param_bounds = np.array([None, None]).reshape(1,2)

    def __init__(self, filter_labels, arf, exposure, redshift,
                 lum_dist=None, cosmology=None,
                 path_to_models=None, path_to_filters=None, **kwargs):
        '''
            Generic initialization. Actual model-building should be handled
            by implementing the `construct_model` and `construct_model_grid` methods.
        '''

        self.redshift = redshift
        self.filter_labels = filter_labels

        # We need a luminosity distance to calculate count rates, etc.
        if (lum_dist is None):

            assert (isinstance(cosmology, FlatLambdaCDM)), "'cosmology' should be provided as an astropy cosmology object"

            lum_dist = cosmology.luminosity_distance(self.redshift).value

            if (lum_dist < 10):
                warnings.warn('Redshift results in a luminosity distance less than 10 Mpc. Setting DL to 10 Mpc to convert flux/uncertainty to Lnu.',
                               AstropyUserWarning)
                lum_dist = 10

        self.DL = lum_dist
        self._DL_cm = (self.DL * u.Mpc).to(u.cm).value

        if (path_to_models is None):
            self.path_to_models = str(Path(__file__).parent.resolve()) + '/../models/'
        else:
            self.path_to_models = path_to_models
            if(self.path_to_models[-1] != '/'): self.path_to_models = self.path_to_models + '/'


        if (path_to_filters is None):
            self.path_to_filters = str(Path(__file__).parent.resolve()) + '/../filters/'
        else:
            self.path_to_filters = path_to_filters
            if(self.path_to_filters[-1] != '/'): self.path_to_filters = self.path_to_filters + '/'

        # Build the actual model -- if the model isn't
        # read in from files, _construct_model should still at
        # least define the wavelength grid.
        self._construct_model(**kwargs)
        self.wave_grid_obs = (1 + self.redshift) * self.wave_grid_rest
        hc_um = (const.c * const.h).to(u.keV * u.micron).value
        self.energ_grid_rest = hc_um / self.wave_grid_rest
        self.energ_grid_obs = self.energ_grid_rest / (1 + self.redshift)
        # Observed-frame photon energy for calculating count-rate
        self.phot_energ = (self.energ_grid_obs * u.keV).to(u.Lsun * u.s).value

        # Ingest the ARF and put it on the energy grid we use here.
        specresp = arf['SPECRESP']
        elo = arf['ENERG_LO']
        ehi = arf['ENERG_HI']

        finterp_arf = interp1d(elo, specresp, bounds_error=False, fill_value=0.0)
        specresp_interp = finterp_arf(self.energ_grid_obs)
        self.specresp = specresp_interp

        # Allow for non-uniform specification of exposure time by
        # letting exposure be either a scalar or an array
        if (isinstance(exposure, numbers.Number)):
            exposure = np.full(len(filter_labels), exposure, dtype='float')
        else:
            assert(len(exposure) == len(filter_labels)), "Exposure time should be either a scalar or an array with one element per bandpass."
            exposure = np.array(exposure, dtype='float')

        self.exposure = exposure

        # Get filters
        self.filters = dict()
        for name in self.filter_labels:
            self.filters[name] = np.zeros(len(self.wave_grid_rest), dtype='float')

        self._get_filters()
        self.Nfilters = len(self.filters)

        c_um = const.c.to(u.micron / u.s).value

        self.wave_obs = np.zeros(len(self.filter_labels), dtype='float')
        self._get_wave_obs()
        self.energ_obs = hc_um / self.wave_obs
        self.nu_obs = c_um / self.wave_obs

        self.nu_grid_rest = c_um / self.wave_grid_rest
        self.nu_grid_obs = c_um / self.wave_grid_obs

        # Extra per-model code to execute once everything is loaded. Usually
        # only necessary for gridded models, hence the name.
        if (self.gridded):

            self._construct_model_grid()

    def _construct_model(self, wave_grid=(1e-6, 1e-1, 200), **kwargs):
        '''
            Build the model from the appropriate files, a function, whatever.
            Here, this just defines the basics.
        '''

        if(isinstance(wave_grid, tuple)):
            self.wave_grid_rest = np.logspace(np.log10(wave_grid[0]), np.log10(wave_grid[1]), wave_grid[2])
        elif(wave_grid is None):
            self.wave_grid_rest = np.logspace(-6, -1, 200)
        else:
            self.wave_grid_rest = wave_grid

        #self.wave_grid_rest = np.logspace(-3, -1, 200)

    def _get_filters(self):
        '''
            Load the filters.
        '''

        self.filters = get_filters(self.filter_labels, self.wave_grid_obs, self.path_to_filters)

    def get_model_countrate_hires(self, params):
        '''
            Produce the high-resolution model countrate density in counts s-1 Hz-1.
            Note that these could probably be implemented here; the only thing that's
            specific to each individual model is the high-resolution Lnu spectrum.
        '''

        raise NotImplementedError('Implemented by child class.')

    def get_model_countrate(self, params):
        '''
            Produce the mean model countrate density in the bandpass.
        '''

        raise NotImplementedError('Implemented by child class.')

    def get_model_counts(self, params):
        '''
            Produce the model counts in the bandpass.
        '''

        raise NotImplementedError('Implemented by child class.')
