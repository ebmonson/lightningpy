import numpy as np
from scipy.integrate import trapz
from scipy.interpolate import interp1d
import astropy.constants as const
import astropy.units as u

from ..base import BaseEmissionModel

__all__ = ['Graybody']

class Graybody(BaseEmissionModel):
    r'''A gray-body dust emission model.

    The gray body is a modified blackbody accounting for variations
    in opacity and emissivity, such that (Casey et al. 2012):

    .. math::

        L_\nu \propto [1 - \exp(-\tau(\nu))] B_\nu(T) = [1 - \exp(-\tau(\nu))] \frac{\nu^3}{\exp(h \nu / k T) - 1}

    where the optical depth is taken to be a power law in frequency: ``tau(nu) = (nu / nu0)^beta``
    for some ``nu0`` where the optical depth is 1. In practice this is usually assumed to be around 100-200 um.
    Beta is expected to range between ~1--2.5.

    Parameters
    ----------
    filter_labels : list, str
        List of filter labels.
    redshift : float
        Redshift of the model.
    wave_grid : np.ndarray
        Rest frame wavelength grid to evaluate the model on.

    '''

    Nparams = 3
    model_name = 'GrayBody-Dust'
    model_type = 'Dust-Emission'
    gridded = False
    param_names = ['gb_lam0', 'gb_beta', 'gb_T']
    param_descr = ['Wavelength of unit optical depth',
                   'Optical depth power law index',
                   'Dust temperature']
    param_names_fncy = [r'$\lambda_0$', r'$\beta$', r'$T$']
    param_bounds = np.array([[10, 1000],
                             [0.1, 5.0],
                             [10, 100]])

    def _construct_model(self, wave_grid=None):
        '''
        Not much to do here.
        '''

        c_um = const.c.to(u.micron / u.s).value

        if wave_grid is not None:

            nu_grid = c_um / wave_grid
            self.wave_grid_rest = wave_grid
            self.wave_grid_obs = self.wave_grid_rest * (1 + self.redshift)
            self.nu_grid_rest = nu_grid
            self.nu_grid_obs = self.nu_grid_rest * (1 + self.redshift)

        else:

            self.wave_grid_rest = np.logspace(0, 3, 150)
            self.wave_grid_obs = self.wave_grid_rest * (1 + self.redshift)
            self.nu_grid_rest = c_um / self.wave_grid_rest
            self.nu_grid_obs = self.nu_grid_rest * (1 + self.redshift)

    def get_model_lnu_hires(self, params):
        '''Construct the high-resolution dust SED.

        Note that the model Lnu is normalized to the total luminosity.

        Parameters
        ----------
        params : np.ndarray, (Nmodels, 3) or (3,) float32
            The dust model parameters.

        Returns
        -------
        Lnu_obs : np.ndarray, (Nmodels, Nwave), (Nmodels, Nages, Nwave), or (Nwave,), float32
            The dust spectrum as seen through the given filters
        Lbol : np.ndarray, (Nmodels,) or (Nmodels, Nages)
            The total luminosity of the dust model.

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

        l0 = params[:,0]
        beta = params[:,1]
        T = params[:,2]

        c_um = 2.9979258e+14 # um s-1
        hcoverk = 1.4388e4 # um K

        # This is a scheme for avoiding overflow at small wavelengths;
        # at the temperatures we allow we should expect no emission
        # at these wavelengths anyway
        Lnu_unnorm = np.zeros((Nmodels, len(self.wave_grid_rest)))
        wave_g1um = self.wave_grid_rest[self.wave_grid_rest >= 1]

        Lnu_unnorm[:, self.wave_grid_rest >= 1] = (1 - np.exp(-1 * (l0[:,None] / wave_g1um[None,:])**beta[:,None])) * \
                                                  (c_um / wave_g1um[None,:]) ** 3 / \
                                                  (np.exp(hcoverk / (wave_g1um[None,:] * T[:,None])) - 1)

        L_total = -1 * trapz(Lnu_unnorm, self.nu_grid_rest, axis=1)

        Lnu_norm = (1 + self.redshift) * Lnu_unnorm / L_total[:, None]

        return Lnu_norm

    def get_model_lnu(self, params):
        '''Construct the dust SED as observed in the given filters.

        Note that the model Lnu is normalized to the total luminosity.

        Parameters
        ----------
        params : np.ndarray, (Nmodels, 3) or (3,) float32
            The dust model parameters.

        Returns
        -------
        Lnu_obs : np.ndarray, (Nmodels, Nfilters), (Nmodels, Nages, Nfilters), or (Nfilters,), float32
            The dust spectrum as seen through the given filters
        Lbol : np.ndarray, (Nmodels,) or (Nmodels, Nages)
            The total luminosity of the dust model.

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

        lnu_hires = self.get_model_lnu_hires(params)

        if (Nmodels == 1):
            lnu_hires = lnu_hires.reshape(1,-1)

        lmod = np.zeros((Nmodels, self.Nfilters))

        for i, filter_label in enumerate(self.filters):
            # Recall that the filters are normalized to 1 when integrated against wave_grid_obs
            # so we can integrate with that to get the mean Lnu in each band.
            lmod[:,i] = trapz(self.filters[filter_label][None,:] * lnu_hires, self.wave_grid_obs, axis=1)

        return lmod
