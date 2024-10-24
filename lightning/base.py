
from pathlib import Path
from importlib.resources import files
import numpy as np

from scipy.integrate import trapezoid as trapz

import astropy.constants as const
import astropy.units as u

from .get_filters import get_filters

class BaseEmissionModel():

    Nparams = 0
    model_name = 'None'
    model_type = 'None'
    gridded = False
    param_names = ['None']
    param_descr = ['None']
    param_names_fncy = [r'None']
    param_bounds = np.array([None, None]).reshape(1,2)

    def __init__(self, filter_labels, redshift, **kwargs):
        '''
            Generic initialization. Actual model-building should be handled by implementing the `construct_model` and
            `construct_model_grid` methods.
        '''

        self.redshift = redshift
        self.filter_labels = filter_labels

        self.modeldir = files('lightning.data.models')

        # Build the actual model
        self._construct_model(**kwargs)
        self.wave_grid_obs = (1 + self.redshift) * self.wave_grid_rest

        # Get filters
        self.filters = dict()
        for name in self.filter_labels:
            self.filters[name] = np.zeros(len(self.wave_grid_rest), dtype='float')

        self._get_filters()
        self.Nfilters = len(self.filters)

        c_um = const.c.to(u.micron / u.s).value

        self.wave_obs = np.zeros(len(self.filter_labels), dtype='float')
        self._get_wave_obs()
        self.nu_obs = c_um / self.wave_obs

        self.nu_grid_rest = c_um / self.wave_grid_rest
        self.nu_grid_obs = c_um / self.wave_grid_obs

        # Extra per-model code to execute once everything is loaded. Usually
        # only necessary for gridded models, hence the name.
        if (self.gridded):

            self._construct_model_grid()


    def _construct_model(self, **kwargs):
        '''
            Build the model from the appropriate files, a function, whatever.
            Here, this just defines a default rest-frame wavelength grid.
        '''

        self.wave_grid_rest = np.logspace(-1, 3, 1200)
        self.Lnu_rest = 0 * self.wave_grid_rest
        self.Lnu_obs = (1 + self.redshift) * self.Lnu_rest

    def _construct_model_grid(self, **kwargs):
        '''
            For some of the models we'll need to arrange the constructed models
            into N-D arrays for later interpolation.
        '''

        raise NotImplementedError('Implemented by child class.')

    def _get_filters(self):
        '''
            Load the filters.
        '''

        self.filters = get_filters(self.filter_labels, self.wave_grid_obs)

    def _get_wave_obs(self):
        '''
            Compute the mean wavelength of the normalized filters.
        '''

        for i, label in enumerate(self.filter_labels):
            lam = trapz(self.wave_grid_obs * self.filters[label], self.wave_grid_obs)
            self.wave_obs[i] = lam

    def _check_bounds(self, params):
        '''
            Check that the parameters are within the ranges where the model is
            meaningful and defined. Return the indices where the model
            is out of bounds.
        '''

        if (len(params.shape) == 1) and (self.Nparams > 1):
            params = params.reshape(1,-1)
        elif (len(params.shape) == 1) and (self.Nparams == 1):
            params = params.reshape(-1,1)

        if self.Nparams is not None:
            assert (self.Nparams == params.shape[1]), "Number of parameters must match the number (%d) expected by this model (%s)" % (self.Nparams, self.model_name)
        Nmodels = params.shape[0]

        ob_idcs = (params < self.param_bounds[:,0][None,:]) | (params > self.param_bounds[:,1][None,:])

        return ob_idcs

    def print_params(self, verbose=False):

        '''
            If `verbose`, print a nicely formatted table
            of the models, their parameters, and the description
            of the parameters.
            Otherwise, just print the names of the parameters.
        '''

        if (verbose):

            from astropy.table import Table
            from astropy.io import ascii

            print('')
            print('============================')
            print(self.model_name)
            print('============================')
            mod_table = Table()
            mod_table['Parameter'] = self.param_names
            mod_table['Lo'] = self.param_bounds[:,0]
            mod_table['Hi'] = self.param_bounds[:,1]
            mod_table['Description'] = self.param_descr
            #print(mod_table)
            ascii.write(mod_table, format='fixed_width_two_line')

        else:

            print(self.param_names)

        print('')
        print('Total parameters: %d' % (self.Nparams))

    def get_model_lnu_hires(self, params):
        '''
            Overwrite this method to provide the actual evaluation of the
            high-res spectral model.
        '''

        raise NotImplementedError('Implemented by child class.')

    def get_model_lnu(self, params):
        '''
            Overwrite this method to provide the actual evaluation of the
            model as observed in the given filters.
        '''

        raise NotImplementedError('Implemented by child class.')
