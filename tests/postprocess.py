'''
postprocess.py

Test the posprocessing methods on the
two fits to the simulated galaxy with
the PEGASE and BPASS stellar populations.

'''

import h5py
import numpy as np
from lightning.postprocessing import postprocess_catalog

def main():

    config_files = ['lgh_bpass.pkl', 'lgh_pegase.pkl', 'lgh_burst.pkl', 'lgh_agn.pkl']
    chain_files = ['chain_bpass.h5', 'chain_pegase.h5', 'chain_burst.h5', 'chain_agn.h5']

    postprocess_catalog(chain_files,
                        config_files,
                        solver_mode='mcmc',
                        model_mode='pickle',
                        names=['bpass', 'pegase', 'burst', 'agn'],
                        catalog_name='catalog.h5')

    # Read in the catalog and print the quantiles for a few quantities,
    # just to make sure everything makes sense.
    f = h5py.File('catalog.h5')
    print('Stellar masses from postprocessed MCMC catalog:')
    print('BPASS Mstar = %.2e (+%.2e) (-%.2e)' % (f['bpass/properties/mstar/med'][()], f['bpass/properties/mstar/hi'][()] - f['bpass/properties/mstar/med'][()], f['bpass/properties/mstar/med'][()] - f['bpass/properties/mstar/lo'][()]))
    print('PEGASE Mstar = %.2e (+%.2e) (-%.2e)' % (f['pegase/properties/mstar/med'][()], f['pegase/properties/mstar/hi'][()] - f['pegase/properties/mstar/med'][()], f['pegase/properties/mstar/med'][()] - f['pegase/properties/mstar/lo'][()]))
    print('Burst Mstar = %.2e (+%.2e) (-%.2e)' % (f['burst/properties/mstar/med'][()], f['burst/properties/mstar/hi'][()] - f['burst/properties/mstar/med'][()], f['burst/properties/mstar/med'][()] - f['burst/properties/mstar/lo'][()]))
    print('AGN Mstar = %.2e (+%.2e) (-%.2e)' % (f['agn/properties/mstar/med'][()], f['agn/properties/mstar/hi'][()] - f['agn/properties/mstar/med'][()], f['agn/properties/mstar/med'][()] - f['agn/properties/mstar/lo'][()]))


    f.close()

    postprocess_catalog(chain_files,
                        config_files,
                        solver_mode='mle',
                        model_mode='pickle',
                        names=['bpass', 'pegase', 'burst', 'agn'],
                        catalog_name='catalog_mle.h5')

    # Read in the catalog and print the quantiles for a few quantities,
    # just to make sure everything makes sense.
    f = h5py.File('catalog_mle.h5')
    print('Stellar masses from postprocessed BFGS catalog:')
    print('BPASS Mstar = %.2e' % (f['bpass/properties/mstar/best'][()]))
    print('PEGASE Mstar = %.2e' % (f['pegase/properties/mstar/best'][()]))
    print('Burst Mstar = %.2e' % (f['burst/properties/mstar/best'][()]))
    print('AGN Mstar = %.2e' % (f['agn/properties/mstar/best'][()]))

if __name__ == '__main__':
    main()
