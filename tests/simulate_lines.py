#!/usr/bin/env python

'''
Simulate line intensities for a variety of SFH and extinction.
'''

import warnings
import pickle
import h5py
import numpy as np
from lightning import Lightning
from lightning.stellar import BPASSModelA24, PEGASEModelA24
from lightning.sfh import PiecewiseConstSFH
from astropy.utils.exceptions import AstropyUserWarning
import astropy.units as u
from astropy.table import Table
from astropy.io import ascii

def main():

    filter_labels = ['SDSS_u', 'SDSS_g', 'SDSS_r', 'SDSS_i', 'SDSS_z']
    line_labels = 'default' # Make sure the default line list is accessible
    # Pretend that I really really care about only the Hydrogen emission
    # line_labels = ['H__1_102572A', 'H__1_121567A', 'H__1_410173A', 'H__1_434046A',
    #                'H__1_486132A','H__1_656280A','H__1_954597A','H__1_187510M','H__1_128181M',
    #                'H__1_109381M', 'H__1_100494M', 'H__1_405115M', 'H__1_262515M', 'H__1_216553M', 'H__1_194456M']

    age_bins = [0.0] + list(np.logspace(7, np.log10(13.4e9), 7))
    stars = BPASSModelA24(filter_labels,
                          0.0,
                          age=age_bins,
                          step=True,
                          nebula_old=False,
                          line_labels=line_labels)

    sfh = PiecewiseConstSFH(age_bins)

    sfh_params =  np.array([1,1,1,0,0,0,0]).reshape(1,-1)
    params = np.array([0.02, -1.5]).reshape(1,-1)

    lines_ext, lines = stars.get_model_lines(sfh, sfh_params, params)

    print('BPASS:')
    for name, L in zip(stars.line_labels, lines.flatten()):
        print('%16s    %.3e' % (name, L))

    stars = PEGASEModelA24(filter_labels,
                           0.0,
                           age=age_bins,
                           step=True,
                           nebula_old=False,
                           line_labels=line_labels)

    lines_ext, lines = stars.get_model_lines(sfh, sfh_params, params)

    print('PEGASE:')
    for name, L, Le in zip(stars.line_labels, lines.flatten(), lines_ext.flatten()):
        print('%16s    %.3e    %.3e' % (name, L, Le))

    lgh = Lightning(filter_labels,
                    redshift=0.0,
                    ages=age_bins,
                    stellar_type='PEGASE-A24',
                    line_labels=line_labels
                    )

    # lgh.print_params(verbose=True)
    params = np.array([[1,1,1,0,0,0,0,0.02,-1.5,0.1,0.0,0.0],
                       [1,1,1,0,0,0,0,0.02,-1.5,0.3,0.0,0.0]])

    lines_ext, lines = lgh.get_model_lines(params)

    print('Lightning:')
    for name, L, Le, Le2 in zip(lgh.stars.line_labels, lines[0,:].flatten(), lines_ext[0,:].flatten(), lines_ext[1,:].flatten()):
        print('%16s    %.3e    %.3e    %.3e' % (name, L, Le, Le2))

    # And now, the burst models
    lgh = Lightning(filter_labels,
                    redshift=0.0,
                    stellar_type='BPASS-A24',
                    SFH_type='Burst',
                    line_labels=line_labels
                    )
    params = np.array([[6, 6, # Log M and log t
                        0.02, -1.5,
                        0.3, 0.0, 0.0]])

    lines_ext, lines = lgh.get_model_lines(params)
    print('Lightning (BPASS inst. burst):')
    for name, L, Le in zip(lgh.stars.line_labels, lines.flatten(), lines_ext.flatten()):
        print('%16s    %.3e    %.3e' % (name, L, Le))

    lgh = Lightning(filter_labels,
                    redshift=0.0,
                    stellar_type='PEGASE-A24',
                    SFH_type='Burst',
                    line_labels=line_labels
                    )
    params = np.array([[6, 6, # Log M and log t
                        0.02, -1.5,
                        0.3, 0.0, 0.0]])

    lines_ext, lines = lgh.get_model_lines(params)
    print('Lightning (PEGASE inst. burst):')
    for name, L, Le in zip(lgh.stars.line_labels, lines.flatten(), lines_ext.flatten()):
        print('%16s    %.3e    %.3e' % (name, L, Le))

    d = 10 * u.Mpc
    LtoF = 1 / (4 * np.pi * d**2)

    line_flux = ((lines_ext.flatten() * u.Lsun) * LtoF).to(u.erg / u.cm**2 / u.s).value

    line_flux_unc = 0.10 * line_flux.flatten()

    lgh = Lightning(filter_labels,
                    redshift=0.0,
                    stellar_type='PEGASE-A24',
                    SFH_type='Burst',
                    line_labels=line_labels,
                    line_flux=line_flux,
                    line_flux_unc=line_flux_unc
                    )

    print('Line luminosities roundtripped through Lightning')
    print('Should be same as the extinguished luminosities above:')
    for name, L, sL in zip(lgh.stars.line_labels, lgh.L_lines, lgh.L_lines_unc):
        print('%16s    %.3e    %.3e' % (name, L, sL))


if __name__ == '__main__':
    main()
