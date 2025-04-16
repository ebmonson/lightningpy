from lightning import Lightning
import numpy as np

class TestSimulator:

    filter_labels = ['GALEX_FUV', 'GALEX_NUV',
                     'SDSS_u', 'SDSS_g', 'SDSS_r', 'SDSS_i', 'SDSS_z',
                     '2MASS_J', '2MASS_H', '2MASS_Ks',
                     'IRAC_CH1', 'IRAC_CH2', 'IRAC_CH3', 'IRAC_CH4',
                     'MIPS_CH1',
                     'PACS_green', 'PACS_red',
                     'SPIRE_250']
    scal = [0.15, 0.15, # GALEX
            0.05, 0.05, 0.05, 0.05, 0.05, # SDSS
            0.10, 0.10, 0.10, # 2MASS JHKs
            0.05, 0.05, 0.05, 0.05, # IRAC
            0.05, # MIPS 24
            0.10, 0.20, # PACS
            0.15] # and SPIRE
    scal = np.array(scal)

    # rng = np.random.default_rng(1234)
    rng = np.random.default_rng()

    d = 8

    def test_filters(self):

        lgh = Lightning(self.filter_labels,
                        lum_dist=self.d)
        
        assert len(lgh.filters) == len(self.filter_labels)
        assert lgh.filter_labels == self.filter_labels



    # def test_bpass_sim(self):
    #     lgh_bp = Lightning(self.filter_labels,
    #                    lum_dist=self.d,
    #                    stellar_type='BPASS-A24',
    #                    atten_type='Modified-Calzetti',
    #                    dust_emission=True,
    #                    SFH_type='Piecewise-Constant',
    #                    model_unc=0.10
    #                    )

    #     params = np.array([1,1,1,1,1,
    #                     0.02, -2.0,
    #                     0.1, -0.1, 0.0,
    #                     2, 1, 3e5, 0.1, 0.025])

    #     Lsim_bp, Lsim_bp_intr = lgh_bp.get_model_lnu(params)

    #     Lsim_bp_unc = self.scal * Lsim_bp
    #     Lsim_bp += self.rng.normal(loc=0, scale=Lsim_bp_unc)

    #     fnu2lnu_gal = 1 / (4 * np.pi * (self.d * u.Mpc) ** 2)
        
    #     fsim_bp = (fnu2lnu_gal * (Lsim_bp * const.Lsun / u.Hz)).to(u.mJy).value
    #     fsim_bp_unc = (fnu2lnu_gal * (Lsim_bp_unc * const.Lsun / u.Hz)).to(u.mJy).value

    #     lgh_bp.flux_obs = fsim_bp
    #     lgh_bp.flux_unc = fsim_bp_unc
    #     lgh_bp.save_pickle('lgh_bpass.pkl')

    #     with h5py.File('simulations_bp.h5', 'w') as f:

    #         f.create_dataset('filter_labels', data=self.filter_labels)
    #         f.create_dataset('normal/BPASS/fnu', data=fsim_bp)
    #         f.create_dataset('normal/BPASS/fnu_unc', data=fsim_bp_unc)
    #         f.create_dataset('normal/BPASS/truth', data=params)
    #         f.create_dataset('normal/BPASS/lumdist', data=self.d)

    # def test_pegase_sim(self):
    #     lgh_pg = Lightning(self.filter_labels,
    #                    lum_dist=self.d,
    #                    stellar_type='PEGASE-A24',
    #                    atten_type='Modified-Calzetti',
    #                    dust_emission=True,
    #                    SFH_type='Piecewise-Constant',
    #                    model_unc=0.10
    #                    )

    #     params = np.array([1,1,1,1,1,
    #                     0.02, -2.0,
    #                     0.1, -0.1, 0.0,
    #                     2, 1, 3e5, 0.1, 0.025])

    #     Lsim_pg, Lsim_pg_intr = lgh_pg.get_model_lnu(params)

    #     Lsim_pg_unc = self.scal * Lsim_pg
    #     Lsim_pg += self.rng.normal(loc=0, scale=Lsim_pg_unc)

    #     fnu2lnu_gal = 1 / (4 * np.pi * (self.d * u.Mpc) ** 2)
        
    #     fsim_pg = (fnu2lnu_gal * (Lsim_pg * const.Lsun / u.Hz)).to(u.mJy).value
    #     fsim_pg_unc = (fnu2lnu_gal * (Lsim_pg_unc * const.Lsun / u.Hz)).to(u.mJy).value

    #     lgh_pg.flux_obs = fsim_pg
    #     lgh_pg.flux_unc = fsim_pg_unc
    #     lgh_pg.save_pickle('lgh_pegase.pkl')

    #     with h5py.File('simulations_pg.h5', 'w') as f:

    #         f.create_dataset('filter_labels', data=self.filter_labels)
    #         f.create_dataset('normal/PEGASE/fnu', data=fsim_pg)
    #         f.create_dataset('normal/PEGASE/fnu_unc', data=fsim_pg_unc)
    #         f.create_dataset('normal/PEGASE/truth', data=params)
    #         f.create_dataset('normal/PEGASE/lumdist', data=self.d)
