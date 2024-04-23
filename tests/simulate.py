'''
simulate.py

Fake four datasets:
- Normal, whole, nearby galaxy, BPASS SSPs
- Normal, whole, nearby galaxy, PEGASE SSPs
- Single, nearby, burst of star formation
- AGN at z = 1, PEGASE SSPs
'''

import h5py
import numpy as np
from lightning import Lightning
import astropy.units as u
import astropy.units as const

def main():

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

    d = 8 # Mpc

    print('...normal galaxy, BPASS')
    lgh_bp = Lightning(filter_labels,
                       lum_dist=d,
                       stellar_type='BPASS-A24',
                       atten_type='Modified-Calzetti',
                       dust_emission=True,
                       SFH_type='Piecewise-Constant',
                       model_unc=0.10
                       )

    params = np.array([1,1,1,1,1,
                       0.02, -2.0,
                       0.1, -0.1, 0.0,
                       2, 1, 3e5, 0.1, 0.025])

    Lsim_bp, Lsim_bp_intr = lgh_bp.get_model_lnu(params)
    lgh_bp.save_pickle('lgh_bpass.pkl')

    print('...normal galaxy, PEGASE')
    lgh_pg = Lightning(filter_labels,
                       lum_dist=d,
                       stellar_type='PEGASE-A24',
                       atten_type='Modified-Calzetti',
                       dust_emission=True,
                       SFH_type='Piecewise-Constant',
                       model_unc=0.10
                       )

    Lsim_pg, Lsim_pg_intr = lgh_pg.get_model_lnu(params)
    lgh_pg.save_pickle('lgh_pegase.pkl')

    print('...single-age burst, BPASS')
    lgh_burst = Lightning(filter_labels,
                          lum_dist=d,
                          stellar_type='BPASS-A24',
                          atten_type='Calzetti',
                          dust_emission=True,
                          SFH_type='Burst',
                          model_unc=0.10
                          )
    params_burst = np.array([4, 6.3,
                             0.02, -2.0,
                             0.1,
                             2, 1, 3e5, 0.1, 0.025])

    Lsim_burst, Lsim_burst_intr = lgh_burst.get_model_lnu(params_burst)
    lgh_burst.save_pickle('lgh_burst.pkl')

    print('...AGN w/ polar dust')
    lgh_agn = Lightning(filter_labels,
                        redshift=1,
                        stellar_type='PEGASE-A24',
                        atten_type='Calzetti',
                        agn_emission=True,
                        agn_polar_dust=True,
                        dust_emission=True,
                        SFH_type='Piecewise-Constant',
                        model_unc=0.10)
    d_agn = lgh_agn.DL
    lgh_agn.save_pickle('lgh_agn.pkl')

    params_agn = np.array([1,1,1,1,1,
                           0.02, -2.0,
                           0.1,
                           2, 1, 3e5, 0.1, 0.025,
                           11, 0.7, 5, 1])

    Lsim_agn, Lsim_agn_intr = lgh_agn.get_model_lnu(params_agn)

    Lsim_bp_unc = scal * Lsim_bp
    Lsim_pg_unc = scal * Lsim_pg
    Lsim_burst_unc = scal * Lsim_burst
    Lsim_agn_unc = scal * Lsim_agn

    rng = np.random.default_rng(1234)
    Lsim_bp += rng.normal(loc=0, scale=Lsim_bp_unc)
    Lsim_pg += rng.normal(loc=0, scale=Lsim_pg_unc)
    Lsim_burst += rng.normal(loc=0, scale=Lsim_burst_unc)
    Lsim_agn += rng.normal(loc=0, scale=Lsim_agn_unc)

    fnu2lnu_gal = 1 / (4 * np.pi * (d * u.Mpc) ** 2)
    fnu2lnu_agn = 1 / (4 * np.pi * (d * u.Mpc) ** 2)

    fsim_bp = (fnu2lnu_gal * (Lsim_bp * const.Lsun / u.Hz)).to(u.mJy).value
    fsim_bp_unc = (fnu2lnu_gal * (Lsim_bp_unc * const.Lsun / u.Hz)).to(u.mJy).value
    fsim_pg = (fnu2lnu_gal * (Lsim_pg * const.Lsun / u.Hz)).to(u.mJy).value
    fsim_pg_unc = (fnu2lnu_gal * (Lsim_pg_unc * const.Lsun / u.Hz)).to(u.mJy).value
    fsim_burst = (fnu2lnu_gal * (Lsim_burst * const.Lsun / u.Hz)).to(u.mJy).value
    fsim_burst_unc = (fnu2lnu_gal * (Lsim_burst_unc * const.Lsun / u.Hz)).to(u.mJy).value
    fsim_agn = (fnu2lnu_agn * (Lsim_agn * const.Lsun / u.Hz)).to(u.mJy).value
    fsim_agn_unc = (fnu2lnu_agn * (Lsim_agn_unc * const.Lsun / u.Hz)).to(u.mJy).value

    with h5py.File('simulations.h5', 'w') as f:

        f.create_dataset('filter_labels', data=filter_labels)
        f.create_dataset('normal/BPASS/fnu', data=fsim_bp)
        f.create_dataset('normal/BPASS/fnu_unc', data=fsim_bp_unc)
        f.create_dataset('normal/BPASS/truth', data=params)
        f.create_dataset('normal/BPASS/lumdist', data=d)
        f.create_dataset('normal/PEGASE/fnu', data=fsim_pg)
        f.create_dataset('normal/PEGASE/fnu_unc', data=fsim_pg_unc)
        f.create_dataset('normal/PEGASE/truth', data=params)
        f.create_dataset('normal/PEGASE/lumdist', data=d)
        f.create_dataset('normal/burst/fnu', data=fsim_burst)
        f.create_dataset('normal/burst/fnu_unc', data=fsim_burst_unc)
        f.create_dataset('normal/burst/truth', data=params_burst)
        f.create_dataset('normal/burst/lumdist', data=d)
        f.create_dataset('agn/fnu', data=fsim_agn)
        f.create_dataset('agn/fnu_unc', data=fsim_agn_unc)
        f.create_dataset('agn/truth', data=params_agn)
        f.create_dataset('agn/redshift', data=1.0)


if __name__ == '__main__':

    main()
