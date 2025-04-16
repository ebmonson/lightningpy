import pytest
from lightning.xray import AGNPlaw, Qsosed
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from lightning.agn import AGNModel
from astropy.table import Table

class TestXrayAGN:

    filter_labels = [
        'XRAY_2000_7000_eV', 'XRAY_0.5_2.0_keV', 
        'GALEX_FUV', 'GALEX_NUV',
        'SDSS_u', 'SDSS_g', 'SDSS_r', 'SDSS_i', 'SDSS_z'
    ]
    
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    redshift = 0.1
    univ_age = cosmo.age(redshift).value

    wave_grid_default = np.logspace(-6,-1,200)

    arf = Table.read('../examples/photometry/cdfn_near_aimpoint.arf')
    # ages = np.array([0, 1e7, 1e8, 1e9, 5e9, univ_age*1e9])

    agn = AGNModel(
        filter_labels,
        redshift
    )

    @pytest.mark.filterwarnings('ignore:invalid value encountered in divide')
    def test_agn_plaw(self):

        xagn = AGNPlaw(
            self.filter_labels,
            None,
            15.e3,
            self.redshift,
            cosmology=self.cosmo
        )

        xagn_arf = AGNPlaw(
            self.filter_labels,
            self.arf,
            2.e6,
            self.redshift,
            cosmology=self.cosmo
        )

        agn_params = np.array([11.0, 0.7, 7, 0.1]).reshape(1,-1)
        params = np.array([1.8, 0.0]).reshape(1,-1)

        # First one should be able to calculate
        # spectral model but not countrate
        lnu,lnuintr = xagn.get_model_lnu(
            params, 
            self.agn, 
            agn_params, 
        )

        with pytest.raises(AssertionError):
            _ = xagn.get_model_countrate(
                params, 
                self.agn, 
                agn_params, 
            )
            
        lnu2,lnuintr2 = xagn_arf.get_model_lnu(
            params, 
            self.agn, 
            agn_params, 
        )


        cts = xagn_arf.get_model_counts(
            params, 
            self.agn, 
            agn_params, 
        )
        
        assert ((lnu.flatten()[:2] == lnu2.flatten()[:2]).all())
        assert (np.count_nonzero(~np.isnan(cts)) == 2)


    def test_qsosed(self):
        
        xagn = Qsosed(
            self.filter_labels,
            None,
            15.e3,
            self.redshift,
            cosmology=self.cosmo
        )

        xagn_arf = Qsosed(
            self.filter_labels,
            self.arf,
            2.e6,
            self.redshift,
            cosmology=self.cosmo
        )

        params = np.array([8.0, -1.0]).reshape(1,-1)

        # First one should be able to calculate
        # spectral model but not countrate
        lnu,lnuintr = xagn.get_model_lnu(
            params
        )

        with pytest.raises(AssertionError):
            _ = xagn.get_model_countrate(
                params
            )
            
        lnu2,lnuintr2 = xagn_arf.get_model_lnu(
            params
        )


        cts = xagn_arf.get_model_counts(
            params
        )
        
        # Default QSOSED wave grid actually goes up to 100 um due to some choice
        # I made as a grad student, so fluxes are not zero or NaN for lam < 100 um;
        # counts will be zero rather than NaN for E < ~0.1 keV and lam < 100 um.
        assert ((lnu.flatten()[:2] == lnu2.flatten()[:2]).all())
        assert (np.count_nonzero(cts) == 2)