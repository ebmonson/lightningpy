import pytest
from lightning.dust import DL07Dust, Graybody
import numpy as np
from astropy.cosmology import FlatLambdaCDM

class TestDust:

    filter_labels = ['GALEX_FUV', 'GALEX_NUV',
                     'SDSS_u', 'SDSS_g', 'SDSS_r', 'SDSS_i', 'SDSS_z',
                     '2MASS_J', '2MASS_H', '2MASS_Ks',
                     'IRAC_CH1', 'IRAC_CH2', 'IRAC_CH3', 'IRAC_CH4',
                     'MIPS_CH1',
                     'PACS_green', 'PACS_red',
                     'SPIRE_250']
    
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    redshift = 0.1
    univ_age = cosmo.age(redshift).value

    def test_dl07(self):

        with pytest.warns(RuntimeWarning):
            d1 = DL07Dust(self.filter_labels, self.redshift)

        params = np.array([2.0, 1.0, 3.e5, 0.1, 0.025])
        lnu, lbol = d1.get_model_lnu(params)

        assert lnu.size == len(self.filter_labels)
        assert lbol.size == 1

        wave_grid = np.logspace(-2, 3, 500)
        d2 = DL07Dust(self.filter_labels, self.redshift, wave_grid=wave_grid)

    def test_gray(self):

        # Default wavelength grid only from 1-1000 um
        with pytest.warns(RuntimeWarning):
            g1 = Graybody(self.filter_labels, self.redshift)
        # In practice the Graybody model is only
        # invoked with this single set of parameters.
        params = np.array([200, 1.5, 100])
        lnu = g1.get_model_lnu(params)
        assert lnu.size == len(self.filter_labels)

        # Custom wavelength grid
        wave_grid = np.logspace(-2, 3, 500)
        g2 = Graybody(self.filter_labels, self.redshift, wave_grid=wave_grid)