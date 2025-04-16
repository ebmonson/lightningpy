import pytest
from lightning.xray import StellarPlaw
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from lightning import get_filters
from lightning.sfh import PiecewiseConstSFH,DelayedExponentialSFH
from lightning.stellar import PEGASEModelA24
from astropy.table import Table

class TestXrayStellar:

    # Incl. specifying the NuSTAR hard band in a wild way to check unit conversion
    filter_labels = [
        'XRAY_2.418e18_7.254e18_Hz', 'XRAY_2000_7000_eV', 'XRAY_0.5_2.0_keV', 
        'GALEX_FUV', 'GALEX_NUV',
        'SDSS_u', 'SDSS_g', 'SDSS_r', 'SDSS_i', 'SDSS_z'
    ]
    
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    redshift = 0.1
    univ_age = cosmo.age(redshift).value

    wave_grid_default = np.logspace(-6,-1,200)

    arf = Table.read('examples/photometry/cdfn_near_aimpoint.arf')
    ages = np.array([0, 1e7, 1e8, 1e9, 5e9, univ_age*1e9])
    sfh = PiecewiseConstSFH(ages)
    stars = PEGASEModelA24(
        filter_labels, 
        redshift,
        age=ages
    )

    stars_cont = PEGASEModelA24(
        filter_labels, 
        redshift,
        step=False
    )
    sfh_cont = DelayedExponentialSFH(stars_cont.age)


    @pytest.mark.filterwarnings('ignore:invalid value encountered in divide')
    def test_filters(self):
        
        filters = get_filters(
            self.filter_labels,
            self.wave_grid_default
        )

        assert len(filters.keys()) == len(self.filter_labels)

    @pytest.mark.filterwarnings('ignore:invalid value encountered in divide')
    def test_stellar_plaw(self):

        st = StellarPlaw(
            self.filter_labels,
            None,
            15.e3,
            self.redshift,
            cosmology=self.cosmo
        )

        st_arf = StellarPlaw(
            self.filter_labels,
            self.arf,
            2.e6,
            self.redshift,
            cosmology=self.cosmo
        )

        params = np.array([1.8])
        st_params = np.array([0.02, -2.0]).reshape(1,-1)
        sfh_params = np.array([1,1,1,1,1]).reshape(1,-1)

        # First one should be able to calculate
        # spectral model but not countrate
        lnu,lnuintr = st.get_model_lnu(
            params, 
            self.stars, 
            st_params, 
            self.sfh,
            sfh_params
        )

        with pytest.raises(AssertionError):
            _ = st.get_model_countrate(
                params, 
                self.stars, 
                st_params, 
                self.sfh,
                sfh_params
            )
            
        lnu2,lnuintr2 = st_arf.get_model_lnu(
            params, 
            self.stars, 
            st_params, 
            self.sfh,
            sfh_params
        )


        cts = st_arf.get_model_counts(
            params, 
            self.stars, 
            st_params, 
            self.sfh,
            sfh_params
        )
        
        assert ((lnu.flatten()[:3] == lnu2.flatten()[:3]).all())
        assert (np.count_nonzero(~np.isnan(cts)) == 3)

        sfh_params_cont = np.array([1, 1e7]).reshape(1,-1)
        cts2 = st_arf.get_model_counts(
            params, 
            self.stars_cont, 
            st_params, 
            self.sfh_cont,
            sfh_params_cont
        )

        assert (np.count_nonzero(~np.isnan(cts)) == 3)
        






