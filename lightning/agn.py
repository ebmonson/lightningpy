'''
    agn.py

    Class interfaces for UV-IR AGN emission.
    Ported from IDL Lightning.
'''

import numpy as np
from scipy.integrate import trapz
from scipy.interpolate import interp1d, interpn
import astropy.constants as const
import astropy.units as u
from astropy.table import Table

from .base import BaseEmissionModel

__all__ = ['AGNModel']

class AGNModel(BaseEmissionModel):
    '''An implementation of the Stalevski (2016) SKIRTOR models.

    Parameters
    ----------
    filter_labels : list, str
        List of filter labels.
    redshift : float
        Redshift of the model.
    wave_grid : np.ndarray, (Nwave,), float, optional
        If set, the spectra are interpreted to this wavelength grid.

    Attributes
    ----------
    Lnu_rest : numpy.ndarray, (29, 7, 132), float
        High-res spectral model grid. First dimension covers U, the second q_PAH,
        and the third covers wavelength.
    Lnu_obs : numpy.ndarray, (29, 7, 132), float
        ``(1 + redshift) * Lnu_rest``
    mean_Lnu : numpy.ndarray, (29, 7, Nfilters), float
        The ``Lnu_obs`` grid integrated against the filters.
    Lbol : numpy.ndarray, (29, 7), float
        Total luminosity of each model in the grid.
    wave_grid_rest : numpy.ndarray, (132,), float
        Rest-frame wavelength grid for the models.
    wave_grid_obs
    nu_grid_rest
    nu_grid_obs

    '''

    Nparams = 3
    model_name = 'SKIRTOR-AGN'
    model_type = 'AGN-Emission'
    gridded = True
    param_names = ['SKIRTOR_log_L_AGN', 'SKIRTOR_cosi_AGN', 'SKIRTOR_tau_97']
    param_descr = ['Integrated luminosity of the model in log Lsun',
                   'Cosine of the inclination to the line of sight',
                   'Edge-on optical depth of the torus at 9.7 microns']
    param_bounds = np.array([[6, 15],
                             [0, 1],
                             [3, 11]])

    def _construct_model(self, wave_grid=None):
        '''
        Load the models from file. This code is executed before _get_filters.
        '''
        self.path_to_models = self.path_to_models + 'agn/stalevski2016/'

        source_table = Table.read(self.path_to_models + 'SKIRTOR.fits.gz')
        # Columns:
        # ['INCLINATION',
        #  'TAU',
        #  'WAVE_REST',
        #  'NU_REST',
        #  'LNU_TOTAL',
        #  'LNU_DISK_DIRECT',
        #  'LNU_DISK_SCATTERED',
        #  'LNU_DUST_TOTAL',
        #  'LNU_DUST_SCATTERED',
        #  'LNU_TRANSPARENT',
        #  'LNU_INTEGRATED',
        #  'DUST_MASS_TOTAL']

        c_um = const.c.to(u.micron / u.s).value

        wave_model = source_table['WAVE_REST'][0,:].data
        nu_model = c_um / wave_model

        # self.wave_grid_rest = source_table['WAVE_REST'][0,:].data
        # self.nu_grid_rest = source_table['NU_REST'][0,:].data
        # self.wave_grid_obs = (1 + self.redshift) * self.wave_grid_rest
        # self.nu_grid_obs = self.nu_grid_rest / (1 + self.redshift)

        self._inc_vec = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
        self._cosinc_vec = np.cos(np.pi * self._inc_vec / 180.0)
        self._tau_vec = np.array([3, 5, 7, 9, 11])

        self._inc_grid = source_table['INCLINATION'].data.reshape(10,5)
        self._tau_grid = source_table['TAU'].data.reshape(10,5)

        # self.Lnu_rest = source_table['LNU_TOTAL'].data.reshape(10,5,132)
        # self.Lnu_rest_transparent = source_table['LNU_TRANSPARENT'].data.reshape(10,5,132)
        # self.Lnu_rest_integrated = source_table['LNU_INTEGRATED'].data.reshape(10,5,132)

        # self.Lnu_obs = (1 + self.redshift) * self.Lnu_rest
        # self.Lnu_obs_transparent = (1 + self.redshift) * self.Lnu_rest_transparent
        # self.Lnu_obs_integrated = (1 + self.redshift) * self.Lnu_rest_integrated

        lnu_total = source_table['LNU_TOTAL'].data.reshape(10,5,132)
        lnu_transp = source_table['LNU_TRANSPARENT'].data.reshape(10,5,132)
        lnu_int = source_table['LNU_INTEGRATED'].data.reshape(10,5,132)

        if (wave_grid is not None):

            finterp_total = interp1d(wave_model, lnu_total, bounds_error=False, fill_value=0.0, axis=2)
            finterp_transp = interp1d(wave_model, lnu_transp, bounds_error=False, fill_value=0.0, axis=2)
            finterp_int = interp1d(wave_model, lnu_int, bounds_error=False, fill_value=0.0, axis=2)
            lnu_total_interp = finterp_total(wave_grid)
            lnu_transp_interp = finterp_transp(wave_grid)
            lnu_int_interp = finterp_int(wave_grid)
            lnu_total_interp[lnu_total_interp < 0.0] = 0.0
            lnu_transp_interp[lnu_transp_interp < 0.0] = 0.0
            lnu_int_interp[lnu_int_interp < 0.0] = 0.0
            nu_grid = c_um / wave_grid

            self.wave_grid_rest = wave_grid
            self.wave_grid_obs = self.wave_grid_rest * (1 + self.redshift)
            self.nu_grid_rest = nu_grid
            self.nu_grid_obs = self.nu_grid_rest * (1 + self.redshift)

            self.Lnu_rest = lnu_total_interp
            self.Lnu_rest_transparent = lnu_transp_interp
            self.Lnu_rest_integrated = lnu_int_interp

        else:

            self.wave_grid_rest = wave_model
            self.wave_grid_obs = self.wave_grid_rest * (1 + self.redshift)
            self.nu_grid_rest = nu_model
            self.nu_grid_obs = self.nu_grid_rest * (1 + self.redshift)
            self.Lnu_rest = lnu_total
            self.Lnu_rest_transparent = lnu_transp
            self.Lnu_rest_integrated = lnu_int

        self.Lnu_obs = (1 + self.redshift) * self.Lnu_rest
        self.Lnu_obs_transparent = (1 + self.redshift) * self.Lnu_rest_transparent
        self.Lnu_obs_integrated = (1 + self.redshift) * self.Lnu_rest_integrated

        finterp_transp = interp1d(wave_model, lnu_transp, axis=2)
        L2500_grid = finterp_transp(0.25)

        self.L2500_rest = L2500_grid
        self.Lbol = trapz(lnu_transp[:,:,::-1], nu_model[::-1], axis=2)


    def _construct_model_grid(self):
        '''
        This code is executed after _get_filters (and is not required in this case).
        '''
        pass

    def get_model_lnu_hires(self, params, exptau=None, exptau_polar=None):
        '''Produce the high-resolution AGN spectral model.

        Given a set of parameters, produce the AGN spectrum, optionally
        attenuating it with the ISM dust attenuation model and a polar dust
        model, a là CIGALE.

        Parameters
        ----------
        params : np.ndarray, (Nmodels, 3) or (3,) float
            The AGN model parameters.
        exptau : np.ndarray, (Nmodels, Nwave) or (Nwave,) float
            The ISM attenuation curve.
        exptau_polar : np.ndarray, (Nmodels, Nwave) or (Nwave,) float
            The polar dust attenuation curve.

        Returns
        -------
        lnu_hires : np.ndarray, (Nmodels, Nwave) or (Nwave,) float
            The high resolution AGN models.

        '''


        param_shape = params.shape # expecting ndarray(Nmodels, Nparams)
        if (len(param_shape) == 1):
            params = params.reshape(1, params.size)
            param_shape = params.shape

        Nmodels = param_shape[0]
        Nparams_in = param_shape[1]

        assert (self.Nparams == Nparams_in), 'The AGN model has 3 parameters'

        if (exptau_polar is not None):
            raise NotImplementedError("Polar dust extinction not implemented yet.")

        ob = self._check_bounds(params)
        if(np.any(ob)):
            raise ValueError('Given parameters are out of bounds for this model (%s).' % (self.model_name))

        # scipy.interpolate.interpn requires the points in the first interpolation
        # dimension to be strictly ascending, so we invert cosi and the corresponding
        # dim. of the Lnu array.
        lnu_hires = 10 ** interpn((self._cosinc_vec[::-1], self._tau_vec),
                                  np.log10(self.Lnu_obs)[::-1, :, :],
                                  params[:,1:],
                                  method='linear',
                                  bounds_error=False,
                                  fill_value=0.0)

        # Interpolate the inclination-integrated spectrum to calculate the
        # bolometric luminosity
        lnu_integrated_hires = 10 ** interpn((self._cosinc_vec[::-1], self._tau_vec),
                                              np.log10(self.Lnu_obs_integrated)[::-1, :, :],
                                              params[:,1:],
                                              method='linear',
                                              bounds_error=False,
                                              fill_value=0.0)

        # lnu_transparent_hires = 10 ** interpn((self._cosinc_vec[::-1], self._tau_vec),
        #                                       np.log10(self.Lnu_obs_transparent)[::-1, :, :],
        #                                       params[:,1:],
        #                                       method='linear',
        #                                       bounds_error=False,
        #                                       fill_value=0.0)

        Lbol = trapz(lnu_integrated_hires[::-1],
                     self.nu_grid_obs[::-1],
                     axis=1)

        if (exptau is None):
            exptau = np.ones_like(lnu_hires)

        assert (exptau.shape == lnu_hires.shape), "Attenuation array and spectral model have different shapes."

        lnu_hires = 10 ** params[:,0][:, None] * exptau * lnu_hires / Lbol[:,None]

        #print(self.nu_grid_obs[:2])
        # print(Lbol.shape)
        # print(Lbol[:10])
        # print()
        # print(Lbol.shape)
        # print(Lbol[:10])

        return lnu_hires

    def get_model_lnu(self, params, exptau=None, exptau_polar=None):
        '''Produce the AGN model as observed in the given filters.

        Given a set of parameters, produce the observed AGN SED, optionally
        attenuating it with the ISM dust attenuation model and a polar dust
        model, a là CIGALE.

        Parameters
        ----------
        params : np.ndarray, (Nmodels, 3) or (3,) float
            The AGN model parameters.
        exptau : np.ndarray, (Nmodels, Nwave) or (Nwave,) float
            The ISM attenuation curve.
        exptau_polar : np.ndarray, (Nmodels, Nwave) or (Nwave,) float
            The polar dust attenuation curve.

        Returns
        -------
        lnu_hires : np.ndarray, (Nmodels, Nfilters) or (Nfilters,) float
            The high resolution AGN models.

        '''

        param_shape = params.shape # expecting ndarray(Nmodels, Nparams)
        if (len(param_shape) == 1):
            params = params.reshape(1, params.size)
            param_shape = params.shape

        Nmodels = param_shape[0]
        Nparams_in = param_shape[1]

        assert (self.Nparams == Nparams_in), 'The AGN model has 3 parameters'

        ob = self._check_bounds(params)
        if(np.any(ob)):
            raise ValueError('Given parameters are out of bounds for this model (%s).' % (self.model_name))

        lnu_hires = self.get_model_lnu_hires(params, exptau=exptau)

        if (Nmodels == 1):
            lnu_hires = lnu_hires.reshape(1,-1)

        lmod = np.zeros((Nmodels, self.Nfilters))

        for i, filter_label in enumerate(self.filters):
            # Recall that the filters are normalized to 1 when integrated against wave_grid_obs
            # so we can integrate with that to get the mean Lnu in each band.
            lmod[:,i] = trapz(self.filters[filter_label][None,:] * lnu_hires, self.wave_grid_obs, axis=1)

        return lmod
