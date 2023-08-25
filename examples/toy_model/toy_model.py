'''
toy_model.py

This script simulates a galaxy with a simple
SFH and fits it.
'''

from pprint import pprint
import json
import numpy as np
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM

from lightning import Lightning
from lightning.priors import UniformPrior
from lightning.plots import sed_plot_bestfit, sed_plot_delchi, chain_plot, corner_plot

import h5py

filter_labels = ['SDSS_u', 'SDSS_g', 'SDSS_r', 'SDSS_i', 'SDSS_z',
                 'IRAC_CH1', 'IRAC_CH2', 'IRAC_CH3', 'IRAC_CH4']

# It is also possible to specify
# a luminosity distance rather than a redshift.
# In that case Lightning assumes that the object is
# close and sets redshift=0.
redshift = 0.1

lgh = Lightning(filter_labels,
                redshift=redshift,
                SFH_type='Piecewise-Constant',
                atten_type='Calzetti',
                dust_emission=False,
                agn_emission=False,
                xray_mode=None,
                print_setup_time=True)

## Fake some data
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
DL = cosmo.luminosity_distance(redshift)

params = np.array([1,1,0,0,0,
                   0.1])

Lmod, _ = lgh.get_model_lnu(params)

snr = 10.0

rng = np.random.default_rng()

Lsim = rng.normal(loc=Lmod, scale=Lmod / snr)
Lsim_unc = Lsim / snr

# A silly step, converting the data to mJy just so lightning can convert back to Lsun Hz-1
fsim = (Lsim * u.Lsun / u.Hz / (4 * np.pi * DL**2)).to(u.mJy).value
fsim_unc = (Lsim_unc  * u.Lsun / u.Hz / (4 * np.pi * DL**2)).to(u.mJy).value

lgh.flux_obs = fsim
lgh.flux_unc = fsim_unc

## Fit with emcee

# This is the mean for our initialization
p = np.array([1,1,1,1,1,
              0.3])

const_dim = np.array([False, False, False, False, False,
                      False])

priors = [UniformPrior([0, 1e1]), # SFH
          UniformPrior([0, 1e1]),
          UniformPrior([0, 1e1]),
          UniformPrior([0, 1e1]),
          UniformPrior([0, 1e1]),
          UniformPrior([0, 3]) # tauV
          ]

var_dim = ~const_dim

Nwalkers = 64
Nsteps = 20000

# Starting the MCMC in a very small Gaussian ball around the above parameters
p0 = p[None, :] + rng.normal(loc=0, scale=1e-3, size=(Nwalkers, len(p)))
p0[:, const_dim] = p[const_dim]

sampler = lgh.fit(p0,
                  method='emcee',
                  Nwalkers=Nwalkers,
                  Nsteps=Nsteps,
                  priors=priors,
                  const_dim=const_dim)

# There are no constant dimensions in this example, but if
# we had any we would build them into the sample array
# with the `const_dim` and `const_vals` keywords
samples, logprob_samples, t = lgh.get_mcmc_chains(sampler)

with open('toy_model_output.npy', 'wb') as f:
    np.save(f, samples)
    np.save(f, logprob_samples)
    np.save(f, t)

lgh.save_json('toy_model_config.json')
lgh.save_pickle('toy_model_lgh.pickle')

with h5py.File('toy_model_ouput.hdf5', 'w') as f:
    f.create_dataset('mcmc/samples', data=samples)
    f.create_dataset('mcmc/logprob_samples', data=logprob_samples)
    f.create_dataset('mcmc/autocorr', data=t)
