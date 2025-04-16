import pytest
from lightning.stellar import PEGASEModel as oldstellar, PEGASEModelA24 as newstellar, PEGASEBurstA24 as burst
from lightning.sfh import PiecewiseConstSFH, DelayedExponentialSFH
import numpy as np
from astropy.cosmology import FlatLambdaCDM

class TestPEGASE:

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
    
    def test_pegase_binned(self):

        # 5 age bins properly configured
        age = np.array([0, 1.0e7, 1.0e8, 1.0e9, 5e9, self.univ_age * 1e9])
        pg = oldstellar(self.filter_labels, 0.1, age=age)

        assert pg.filter_labels == self.filter_labels
        assert pg.Lnu_obs.shape == (5, len(pg.Zmet), len(pg.wave_grid_obs))
        assert pg.mstar.shape == (5, len(pg.Zmet))
        assert pg.q0.shape == (5, len(pg.Zmet))
        assert pg.Lbol.shape == (5, len(pg.Zmet))
        
        sfh = PiecewiseConstSFH(age)
        lnu,_,_ = pg.get_model_lnu(sfh, np.array([1,1,1,1,1]), np.array([0.02]))
        assert len(lnu) == len(self.filter_labels)
        mstar_coeff = pg.get_mstar_coeff(0.020)
        assert mstar_coeff.size == 5
        lines = pg.get_model_lines(sfh, 
                                     np.array([1,1,1,1,1]).reshape(1,5), 
                                     np.array([0.02]).reshape(1,1))
        assert lines.size == pg.line_names.size

        # Non-default wavelength grid
        wave_grid = np.logspace(-2, 2, 100)
        # This should warn since the new wave grid does not conver PACS or SPIRE.
        with pytest.warns(RuntimeWarning):
            pg2 = oldstellar(self.filter_labels, 0.1, age=age, wave_grid=wave_grid)
        assert pg2.Lnu_obs.shape == (5, len(pg2.Zmet), len(pg2.wave_grid_obs))
        assert np.all(wave_grid == pg2.wave_grid_rest)
        assert np.all(wave_grid * (1 + self.redshift) == pg2.wave_grid_obs)

        # Too old
        age[-1] = 1.1 * self.univ_age * 1e9
        with pytest.raises(AssertionError):
            pg3 = oldstellar(self.filter_labels, 0.1, age=age)

    def test_pegase_continuous(self):
        
        # Using the stellar age specification from the model files
        pg = oldstellar(self.filter_labels, 0.1, age=None, step=False)

        assert pg.filter_labels == self.filter_labels
        assert pg.Lnu_obs.shape == (62, len(pg.Zmet), len(pg.wave_grid_obs))
        assert pg.mstar.shape == (62, len(pg.Zmet))
        assert pg.q0.shape == (62, len(pg.Zmet))
        assert pg.Lbol.shape == (62, len(pg.Zmet))

        sfh = DelayedExponentialSFH(pg.age)
        lnu,_,_ = pg.get_model_lnu(sfh, np.array([1,5e7]), np.array([0.02]))
        assert len(lnu) == len(self.filter_labels)
        mstar_coeff = pg.get_mstar_coeff(0.020)
        assert mstar_coeff.size == 62
        lines = pg.get_model_lines(sfh, np.array([1,5e7]), np.array([0.02]))
        assert lines.size == pg.line_names.size

        # Non-default age grid
        age = np.logspace(6, np.log10(self.univ_age * 1e9), 50)
        age[-1] = self.univ_age * 1e9 
        pg2 = oldstellar(self.filter_labels, 0.1, age=age, step=False)

        assert pg2.filter_labels == self.filter_labels
        assert pg2.Lnu_obs.shape == (50, len(pg2.Zmet), len(pg2.wave_grid_obs))
        assert pg2.mstar.shape == (50, len(pg2.Zmet))
        assert pg2.q0.shape == (50, len(pg2.Zmet))
        assert pg2.Lbol.shape == (50, len(pg2.Zmet))
        assert np.all(pg2.age == age)

    def test_pegase_A24_binned(self):

        # 5 age bins properly configured
        age = np.array([0, 1.0e7, 1.0e8, 1.0e9, 5e9, self.univ_age * 1e9])
        pg = newstellar(self.filter_labels, 0.1, age=age)

        assert pg.filter_labels == self.filter_labels
        assert pg.Lnu_obs.shape == (5, len(pg.Zmet), len(pg.logU), len(pg.wave_grid_obs))
        assert pg.mstar.shape == (5, len(pg.Zmet))
        assert pg.q0.shape == (5, len(pg.Zmet))
        assert pg.Lbol.shape == (5, len(pg.Zmet))

        sfh = PiecewiseConstSFH(age)
        lnu,_,_ = pg.get_model_lnu(sfh, 
                                   np.array([1,1,1,1,1]).reshape(1,5), 
                                   np.array([0.02, -2.0]).reshape(1,2))
        assert len(lnu) == len(self.filter_labels)
        mstar_coeff = pg.get_mstar_coeff(0.020)
        assert mstar_coeff.size == 5
        lines,_ = pg.get_model_lines(sfh, 
                                     np.array([1,1,1,1,1]).reshape(1,5), 
                                     np.array([0.02, -2.0]).reshape(1,2))
        assert lines.size == pg.line_labels.size

        # Non-default wavelength grid
        wave_grid = np.logspace(-2, 2, 100)
        # This should warn since the new wave grid does not conver PACS or SPIRE.
        with pytest.warns(RuntimeWarning):
            pg2 = newstellar(self.filter_labels, 0.1, age=age, wave_grid=wave_grid)
        assert pg2.Lnu_obs.shape == (5, len(pg2.Zmet), len(pg.logU), len(pg2.wave_grid_obs))
        assert np.all(wave_grid == pg2.wave_grid_rest)
        assert np.all(wave_grid * (1 + self.redshift) == pg2.wave_grid_obs)

        # Too old
        age[-1] = 1.1 * self.univ_age * 1e9
        with pytest.raises(AssertionError):
            pg3 = newstellar(self.filter_labels, 0.1, age=age)

        # Non-default linelist
        age[-1] = self.univ_age * 1e9
        pg4 = newstellar(self.filter_labels, 0.1, age=age, line_labels='full')
        assert pg4.line_lum.shape == (5, len(pg4.Zmet), len(pg4.logU), len(pg4.line_labels))

        # Extra non-default linelist
        pg5 = newstellar(self.filter_labels, 0.1, age=age, line_labels=['H__1_656280A', 'H__1_486132A'])
        assert pg5.line_lum.shape == (5, len(pg5.Zmet), len(pg5.logU), 2)
        assert np.all(pg5.line_labels == np.array(['H__1_656280A', 'H__1_486132A']))

        # Wrong linelist
        with pytest.raises(AssertionError):
            pg6 = newstellar(self.filter_labels, 0.1, age=age, line_labels=['HA6562', 'HB4861'])

    def test_pegase_A24_continuous(self):
        
        # Using the stellar age specification from the model files
        pg = newstellar(self.filter_labels, 0.1, age=None, step=False)

        assert pg.filter_labels == self.filter_labels
        assert pg.Lnu_obs.shape == (42, len(pg.Zmet), len(pg.logU), len(pg.wave_grid_obs))
        assert pg.mstar.shape == (42, len(pg.Zmet))
        assert pg.q0.shape == (42, len(pg.Zmet))
        assert pg.Lbol.shape == (42, len(pg.Zmet))

        sfh = DelayedExponentialSFH(pg.age)
        lnu,_,_ = pg.get_model_lnu(sfh, 
                                   np.array([1,5e7]).reshape(1,2), 
                                   np.array([0.02, -2.0]).reshape(1,2))
        assert len(lnu) == len(self.filter_labels)
        mstar_coeff = pg.get_mstar_coeff(0.020)
        assert mstar_coeff.size == 42
        lines,_ = pg.get_model_lines(sfh, 
                                     np.array([1,5e7]).reshape(1,2), 
                                     np.array([0.02,-2.0]).reshape(1,2))
        assert lines.size == pg.line_labels.size

        # Non-default age grid
        age = np.logspace(6, np.log10(self.univ_age * 1e9), 50)
        age[-1] = self.univ_age * 1e9 
        pg2 = newstellar(self.filter_labels, 0.1, age=age, step=False)

        assert pg2.filter_labels == self.filter_labels
        assert pg2.Lnu_obs.shape == (50, len(pg2.Zmet), len(pg.logU), len(pg2.wave_grid_obs))
        assert pg2.mstar.shape == (50, len(pg2.Zmet))
        assert pg2.q0.shape == (50, len(pg2.Zmet))
        assert pg2.Lbol.shape == (50, len(pg2.Zmet))
        assert np.all(pg2.age == age)

    def test_pegase_A24_burst(self):

        pg = burst(self.filter_labels, self.redshift)

        lnu,_,_ = pg.get_model_lnu(np.array([6,7,0.02,-2.0]).reshape(1,4))
        assert lnu.size == len(self.filter_labels)
        lines,_ = pg.get_model_lines(np.array([6,7,0.02,-2.0]).reshape(1,4))
        assert lines.size == pg.line_labels.size



        