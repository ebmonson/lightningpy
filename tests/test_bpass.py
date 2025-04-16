import pytest
from lightning.stellar import BPASSModel as oldstellar, BPASSModelA24 as newstellar, BPASSBurstA24 as burst
from lightning.sfh import PiecewiseConstSFH, DelayedExponentialSFH
import numpy as np
from astropy.cosmology import FlatLambdaCDM

class TestBPASS:

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
    
    def test_bpass_binned(self):

        # 5 age bins properly configured
        age = np.array([0, 1.0e7, 1.0e8, 1.0e9, 5e9, self.univ_age * 1e9])
        # Should warn due to not covering SPIRE
        with pytest.warns(RuntimeWarning):
            bp = oldstellar(self.filter_labels, 0.1, age=age)

        assert bp.filter_labels == self.filter_labels
        assert bp.Lnu_obs.shape == (5, len(bp.Zmet), len(bp.logU), len(bp.wave_grid_obs))
        assert bp.mstar.shape == (5, len(bp.Zmet))
        assert bp.q0.shape == (5, len(bp.Zmet))
        assert bp.Lbol.shape == (5, len(bp.Zmet))
        
        sfh = PiecewiseConstSFH(age)
        lnu,_,_ = bp.get_model_lnu(sfh, 
                                   np.array([1,1,1,1,1]).reshape(1,5),
                                   np.array([0.02, -2.0]).reshape(1,2))
        assert len(lnu) == len(self.filter_labels)
        mstar_coeff = bp.get_mstar_coeff(0.020)
        assert mstar_coeff.size == 5
        lines = bp.get_model_lines(sfh, 
                                     np.array([1,1,1,1,1]).reshape(1,5), 
                                     np.array([0.02, -2.0]).reshape(1,2))
        assert lines.size == bp.line_names.size

        # Non-default wavelength grid
        wave_grid = np.logspace(-2, 2, 100)
        # This should warn since the new wave grid does not conver PACS or SPIRE.
        with pytest.warns(RuntimeWarning):
            bp2 = oldstellar(self.filter_labels, 0.1, age=age, wave_grid=wave_grid)
        assert bp2.Lnu_obs.shape == (5, len(bp2.Zmet), len(bp.logU), len(bp2.wave_grid_obs))
        assert np.all(wave_grid == bp2.wave_grid_rest)
        assert np.all(wave_grid * (1 + self.redshift) == bp2.wave_grid_obs)

        # Too old
        age[-1] = 1.1 * self.univ_age * 1e9
        with pytest.raises(AssertionError):
            bp3 = oldstellar(self.filter_labels, 0.1, age=age)

    def test_bpass_continuous(self):
        
        # Using the stellar age specification from the model files
        # Should warn due to not covering SPIRE
        with pytest.warns(RuntimeWarning):
            bp = oldstellar(self.filter_labels, 0.1, age=None, step=False)

        assert bp.filter_labels == self.filter_labels
        assert bp.Lnu_obs.shape == (42, len(bp.Zmet), len(bp.logU), len(bp.wave_grid_obs))
        assert bp.mstar.shape == (42, len(bp.Zmet))
        assert bp.q0.shape == (42, len(bp.Zmet))
        assert bp.Lbol.shape == (42, len(bp.Zmet))

        sfh = DelayedExponentialSFH(bp.age)
        lnu,_,_ = bp.get_model_lnu(sfh,
                                   np.array([1,5e7]).reshape(1,2),
                                   np.array([0.02,-2.0]).reshape(1,2))
        assert len(lnu) == len(self.filter_labels)
        mstar_coeff = bp.get_mstar_coeff(0.020)
        assert mstar_coeff.size == 42
        lines = bp.get_model_lines(sfh,
                                   np.array([1,5e7]).reshape(1,2),
                                   np.array([0.02,-2.0]).reshape(1,2))
        assert lines.size == bp.line_names.size

        # Non-default age grid
        age = np.logspace(6, np.log10(self.univ_age * 1e9), 50)
        age[-1] = self.univ_age * 1e9 
        with pytest.warns(RuntimeWarning):
            bp2 = oldstellar(self.filter_labels, 0.1, age=age, step=False)

        assert bp2.filter_labels == self.filter_labels
        assert bp2.Lnu_obs.shape == (50, len(bp2.Zmet), len(bp2.logU), len(bp2.wave_grid_obs))
        assert bp2.mstar.shape == (50, len(bp2.Zmet))
        assert bp2.q0.shape == (50, len(bp2.Zmet))
        assert bp2.Lbol.shape == (50, len(bp2.Zmet))
        assert np.all(bp2.age == age)

    def test_bpass_A24_binned(self):

        # 5 age bins properly configured
        age = np.array([0, 1.0e7, 1.0e8, 1.0e9, 5e9, self.univ_age * 1e9])
        bp = newstellar(self.filter_labels, 0.1, age=age)

        assert bp.filter_labels == self.filter_labels
        assert bp.Lnu_obs.shape == (5, len(bp.Zmet), len(bp.logU), len(bp.wave_grid_obs))
        assert bp.mstar.shape == (5, len(bp.Zmet))
        assert bp.q0.shape == (5, len(bp.Zmet))
        assert bp.Lbol.shape == (5, len(bp.Zmet))

        sfh = PiecewiseConstSFH(age)
        lnu,_,_ = bp.get_model_lnu(sfh, 
                                   np.array([1,1,1,1,1]).reshape(1,5), 
                                   np.array([0.02, -2.0]).reshape(1,2))
        assert len(lnu) == len(self.filter_labels)
        mstar_coeff = bp.get_mstar_coeff(0.020)
        assert mstar_coeff.size == 5
        lines,_ = bp.get_model_lines(sfh, 
                                     np.array([1,1,1,1,1]).reshape(1,5), 
                                     np.array([0.02, -2.0]).reshape(1,2))
        assert lines.size == bp.line_labels.size

        # Non-default wavelength grid
        wave_grid = np.logspace(-2, 2, 100)
        # This should warn since the new wave grid does not conver PACS or SPIRE.
        with pytest.warns(RuntimeWarning):
            bp2 = newstellar(self.filter_labels, 0.1, age=age, wave_grid=wave_grid)
        assert bp2.Lnu_obs.shape == (5, len(bp2.Zmet), len(bp.logU), len(bp2.wave_grid_obs))
        assert np.all(wave_grid == bp2.wave_grid_rest)
        assert np.all(wave_grid * (1 + self.redshift) == bp2.wave_grid_obs)

        # Too old
        age[-1] = 1.1 * self.univ_age * 1e9
        with pytest.raises(AssertionError):
            bp3 = newstellar(self.filter_labels, 0.1, age=age)

        # Non-default linelist
        age[-1] = self.univ_age * 1e9
        bp4 = newstellar(self.filter_labels, 0.1, age=age, line_labels='full')
        assert bp4.line_lum.shape == (5, len(bp4.Zmet), len(bp4.logU), len(bp4.line_labels))

        # Extra non-default linelist
        bp5 = newstellar(self.filter_labels, 0.1, age=age, line_labels=['H__1_656280A', 'H__1_486132A'])
        assert bp5.line_lum.shape == (5, len(bp5.Zmet), len(bp5.logU), 2)
        assert np.all(bp5.line_labels == np.array(['H__1_656280A', 'H__1_486132A']))

        # Wrong linelist
        with pytest.raises(AssertionError):
            bp6 = newstellar(self.filter_labels, 0.1, age=age, line_labels=['HA6562', 'HB4861'])

    def test_bpass_A24_continuous(self):
        
        # Using the stellar age specification from the model files
        bp = newstellar(self.filter_labels, 0.1, age=None, step=False)

        assert bp.filter_labels == self.filter_labels
        assert bp.Lnu_obs.shape == (42, len(bp.Zmet), len(bp.logU), len(bp.wave_grid_obs))
        assert bp.mstar.shape == (42, len(bp.Zmet))
        assert bp.q0.shape == (42, len(bp.Zmet))
        assert bp.Lbol.shape == (42, len(bp.Zmet))

        sfh = DelayedExponentialSFH(bp.age)
        lnu,_,_ = bp.get_model_lnu(sfh, 
                                   np.array([1,5e7]).reshape(1,2), 
                                   np.array([0.02, -2.0]).reshape(1,2))
        assert len(lnu) == len(self.filter_labels)
        mstar_coeff = bp.get_mstar_coeff(0.020)
        assert mstar_coeff.size == 42
        lines,_ = bp.get_model_lines(sfh, 
                                     np.array([1,5e7]).reshape(1,2), 
                                     np.array([0.02,-2.0]).reshape(1,2))
        assert lines.size == bp.line_labels.size

        # Non-default age grid
        age = np.logspace(6, np.log10(self.univ_age * 1e9), 50)
        age[-1] = self.univ_age * 1e9 
        bp2 = newstellar(self.filter_labels, 0.1, age=age, step=False)

        assert bp2.filter_labels == self.filter_labels
        assert bp2.Lnu_obs.shape == (50, len(bp2.Zmet), len(bp.logU), len(bp2.wave_grid_obs))
        assert bp2.mstar.shape == (50, len(bp2.Zmet))
        assert bp2.q0.shape == (50, len(bp2.Zmet))
        assert bp2.Lbol.shape == (50, len(bp2.Zmet))
        assert np.all(bp2.age == age)

    def test_bpass_A24_burst(self):

        bp = burst(self.filter_labels, self.redshift)

        lnu,_,_ = bp.get_model_lnu(np.array([6,7,0.02,-2.0]).reshape(1,4))
        assert lnu.size == len(self.filter_labels)
        lines,_ = bp.get_model_lines(np.array([6,7,0.02,-2.0]).reshape(1,4))
        assert lines.size == bp.line_labels.size



        