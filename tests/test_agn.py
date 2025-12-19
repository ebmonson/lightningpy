from lightning.agn import AGNModel
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

    def test_SKIRTOR(self):

        a1 = AGNModel(self.filter_labels, self.redshift)

        params = np.array([11.0, 0.7, 7, 0.1])
        lnu = a1.get_model_lnu(params)

        assert lnu.size == len(self.filter_labels)

        wave_grid = np.logspace(-2, 3, 500)
        a2 = AGNModel(self.filter_labels, self.redshift, wave_grid=wave_grid)

