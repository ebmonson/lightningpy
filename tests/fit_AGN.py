'''
fit_PEGASE.py

Fit the simulated normal galaxy with PEGASE SSPs.
'''

import warnings
import pickle
import h5py
import numpy as np
from lightning import Lightning
from lightning.priors import UniformPrior, NormalPrior, ConstantPrior
from astropy.utils.exceptions import AstropyUserWarning
from astropy.table import Table
from astropy.io import ascii

def main():

    f = h5py.File('simulations.h5','r')

    with open('lgh_agn.pkl','rb') as p:
        lgh_agn = pickle.load(p)

    lgh_agn.flux_obs = f['agn/fnu'][:]
    lgh_agn.flux_unc = f['agn/fnu_unc'][:]

    priors = 5 * [UniformPrior([0,10])] +\
                 [NormalPrior([0.020, 0.001]), ConstantPrior([-2.0])] +\
                 [UniformPrior([0,3])] +\
                 [ConstantPrior([2]), UniformPrior([0.1,25]), ConstantPrior([3e5]), UniformPrior([0,1]), UniformPrior([0.0047, 0.0458])] +\
                 [UniformPrior([11,13]), UniformPrior([0.5, 1.0]), ConstantPrior([5]), UniformPrior([1,3])]
    priors = np.array(priors)

    Nwalkers = 64
    Nsteps = 2000
    p0s = [pr.sample(Nwalkers) for pr in priors]
    p0 = np.stack(p0s, axis=-1)
    const_dim = np.array([pr.model_name == 'constant' for pr in priors])
    const_vals = np.array([pr.val for pr in priors[const_dim]])

    sampler = lgh_agn.fit(p0,
                         method='emcee',
                         priors=priors,
                         const_dim=const_dim,
                         Nwalkers=Nwalkers,
                         Nsteps=Nsteps,
                         progress=False
                         )
    chain, logprob_chain, tau = lgh_agn.get_mcmc_chains(sampler,
                                                       discard=1000,
                                                       thin=50,
                                                       Nsamples=1000,
                                                       const_dim=const_dim,
                                                       const_vals=const_vals)

    var = np.var(chain, axis=0)

    if np.any((var==0) != const_dim):
        warnings.warn(AstropyUserWarning('Constant dimensions have nonzero variance!'))

    mcmc_med = np.median(chain, axis=0)

    bounds = 5 * [(0,10)] +\
             [(0.020, 0.020), (-2,-2)] +\
             [(0,3)] +\
             [(2,2), (0.1, 25), (3e5, 3e5), (0,1), (0.0047, 0.0458)] +\
             [(11,13), (0.5, 1.0), (5,5), (1,3)]

    p0_lbfgs = p0.copy()
    p0_lbfgs[:,2] = 0.020
    const_dim2 = np.array([b[1] == b[0] for b in bounds])
    const_vals2 = p0_lbfgs[0,const_dim2].flatten()

    # This is usually a pretty terrible starting point, so
    # we may not see convergence.
    lnL = lgh_agn.get_model_likelihood(p0_lbfgs, negative=True)
    best = np.argmin(lnL)
    res = lgh_agn.fit(p0_lbfgs[best,:].flatten(),
                     method='optimize',
                     MCMC_followup=True,
                     force=True,
                     MCMC_kwargs={'Nwalkers':64,'Nsteps':1000,'progress':False, 'init_scale':1e-3},
                     disp=False,
                     bounds=bounds)


    chain2, logprob_chain2, tau2 = lgh_agn.get_mcmc_chains(res[1],
                                                          discard=500,
                                                          thin=25,
                                                          Nsamples=1000,
                                                          const_dim=const_dim2,
                                                          const_vals=const_vals2)
    lbfgs_med = np.median(chain2, axis=0)

    param_names = []
    for mod in lgh_agn.model_components:
        if (mod is not None) and (mod.Nparams != 0):
            param_names += mod.param_names

    with h5py.File('chain_agn.h5', 'w') as f_out:
        f_out.create_dataset('mcmc/samples', data=chain)
        f_out.create_dataset('mcmc/logprob_samples', data=logprob_chain)
        f_out.create_dataset('res/bestfit', data=res[0].x)
        f_out.create_dataset('res/chi2_best', data=res[0].fun)

    t = Table()
    t['PARAM'] = param_names
    t['TRUTH'] = f['agn/truth']
    t['MCMC_MED'] = mcmc_med
    t['LBFGS_BESTFIT'] = res[0].x
    t['LBFGS_MED'] = lbfgs_med
    ascii.write(t, format='fixed_width_two_line')

    f.close()

if __name__ == '__main__':

    main()
