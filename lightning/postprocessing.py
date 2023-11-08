#!/usr/bin/env python

from os import path
import json
import pickle
import numpy as np
from tqdm import tqdm
from astropy.table import Table
from lightning import Lightning
from lightning.ppc import ppc

import h5py

def postprocess_catalog_mcmc(chain_filenames,
                             model_filenames,
                             model_mode='json',
                             names=None,
                             catalog_name='postprocessed_catalog.hdf5'):

    assert (len(chain_filenames) == len(model_filenames)), "We require the same number of chain and model filenames."
    if names is not None:
        assert (len(names) == len(chain_filenames)), "We require the same number of source names as chains."
    else:
        # Assume that anything before the first underscore in the chain files is the name
        names = [path.splitext(path.basename(s))[0].split('_')[0] for s in chain_filenames]

    assert (model_mode in ['json', 'pickle']), "'model_mode' must be either 'json' or 'pickle'."

    with h5py.File(catalog_name, 'w') as outfile:

        for i, cf, mf, n in tqdm(zip(np.arange(len(chain_filenames)), chain_filenames, model_filenames, names), total=len(chain_filenames)):

            with h5py.File(cf, 'r') as f:

                # Writing to in-memory arrays so that samples and logprob_samples don't go
                # out of scope when the file closes. This is probably silly, and contrary to
                # the whole point of HDF5. Could just bump everything under the above context
                # instead.
                samples = np.zeros(f['mcmc/samples'].shape)
                f['mcmc/samples'].read_direct(samples)
                logprob_samples = np.zeros(f['mcmc/logprob_samples'].shape)
                f['mcmc/logprob_samples'].read_direct(logprob_samples)

            source = outfile.create_group(n)

            source.create_dataset('mcmc/samples', data=samples, compression='gzip')
            source.create_dataset('mcmc/logprob_samples', data=logprob_samples, compression='gzip')

            if (model_mode == 'json'):
                lgh = Lightning.from_json(mf)
            elif (model_mode == 'pickle'):
                with open(mf, 'rb') as f:
                    lgh = pickle.load(f)

            bestfit = np.argmax(logprob_samples)

            ##### PARAMETERS
            sfh_params, atten_params, dust_params, agn_params, st_xray_params, agn_xray_params, xray_abs_params = lgh._separate_params(samples)
            parameters = source.create_group('parameters')
            if (sfh_params is not None):
                for j,pname in enumerate(lgh.sfh.param_names):
                    p_quant = np.nanquantile(sfh_params[:,j], q=(0.16, 0.50, 0.84))
                    p_best = sfh_params[bestfit, j]
                    parameters.create_dataset('sfh/%s/lo' % (pname), data=p_quant[0])
                    parameters.create_dataset('sfh/%s/med' % (pname), data=p_quant[1])
                    parameters.create_dataset('sfh/%s/hi' % (pname), data=p_quant[2])
                    parameters.create_dataset('sfh/%s/best' % (pname), data=p_best)
            if (atten_params is not None):
                for j,pname in enumerate(lgh.atten.param_names):
                    p_quant = np.nanquantile(atten_params[:,j], q=(0.16, 0.50, 0.84))
                    p_best = atten_params[bestfit, j]
                    parameters.create_dataset('atten/%s/lo' % (pname), data=p_quant[0])
                    parameters.create_dataset('atten/%s/med' % (pname), data=p_quant[1])
                    parameters.create_dataset('atten/%s/hi' % (pname), data=p_quant[2])
                    parameters.create_dataset('atten/%s/best' % (pname), data=p_best)
            if (dust_params is not None):
                for j,pname in enumerate(lgh.dust.param_names):
                    p_quant = np.nanquantile(dust_params[:,j], q=(0.16, 0.50, 0.84))
                    p_best = dust_params[bestfit, j]
                    parameters.create_dataset('dust/%s/lo' % (pname), data=p_quant[0])
                    parameters.create_dataset('dust/%s/med' % (pname), data=p_quant[1])
                    parameters.create_dataset('dust/%s/hi' % (pname), data=p_quant[2])
                    parameters.create_dataset('dust/%s/best' % (pname), data=p_best)
            if (agn_params is not None):
                for j,pname in enumerate(lgh.agn.param_names):
                    p_quant = np.nanquantile(agn_params[:,j], q=(0.16, 0.50, 0.84))
                    p_best = agn_params[bestfit, j]
                    parameters.create_dataset('agn/%s/lo' % (pname), data=p_quant[0])
                    parameters.create_dataset('agn/%s/med' % (pname), data=p_quant[1])
                    parameters.create_dataset('agn/%s/hi' % (pname), data=p_quant[2])
                    parameters.create_dataset('agn/%s/best' % (pname), data=p_best)
            if (st_xray_params is not None):
                for j,pname in enumerate(lgh.xray_stellar_em.param_names):
                    p_quant = np.nanquantile(st_xray_params[:,j], q=(0.16, 0.50, 0.84))
                    p_best = st_xray_params[bestfit, j]
                    parameters.create_dataset('xray_stellar/%s/lo' % (pname), data=p_quant[0])
                    parameters.create_dataset('xray_stellar/%s/med' % (pname), data=p_quant[1])
                    parameters.create_dataset('xray_stellar/%s/hi' % (pname), data=p_quant[2])
                    parameters.create_dataset('xray_stellar/%s/best' % (pname), data=p_best)
            if (agn_xray_params is not None):
                for j,pname in enumerate(lgh.xray_agn_em.param_names):
                    p_quant = np.nanquantile(agn_xray_params[:,j], q=(0.16, 0.50, 0.84))
                    p_best = agn_xray_params[bestfit, j]
                    parameters.create_dataset('xray_agn/%s/lo' % (pname), data=p_quant[0])
                    parameters.create_dataset('xray_agn/%s/med' % (pname), data=p_quant[1])
                    parameters.create_dataset('xray_agn/%s/hi' % (pname), data=p_quant[2])
                    parameters.create_dataset('xray_agn/%s/best' % (pname), data=p_best)
            if (xray_abs_params is not None):
                for j,pname in enumerate(lgh.xray_abs_intr.param_names):
                    p_quant = np.nanquantile(xray_abs_params[:,j], q=(0.16, 0.50, 0.84))
                    p_best = xray_abs_params[bestfit, j]
                    parameters.create_dataset('xray_abs/%s/lo' % (pname), data=p_quant[0])
                    parameters.create_dataset('xray_abs/%s/med' % (pname), data=p_quant[1])
                    parameters.create_dataset('xray_abs/%s/hi' % (pname), data=p_quant[2])
                    parameters.create_dataset('xray_abs/%s/best' % (pname), data=p_best)

            ##### PROPERTIES
            properties = source.create_group('properties')
            if (lgh.sfh.type == 'piecewise'):
                mstar = lgh.sfh.sum(sfh_params, lgh.stars.mstar)
            else:
                mstar = lgh.sfh.integrate(sfh_params, lgh.stars.mstar)

            mstar_q = np.nanquantile(mstar, q=(0.16, 0.50, 0.84))

            properties.create_dataset('redshift', data=lgh.redshift)
            properties.create_dataset('lumdist', data=lgh.DL)
            properties.create_dataset('filter_labels', data=lgh.filter_labels)
            properties.create_dataset('lnu', data=lgh.Lnu_obs)
            properties.create_dataset('lnu_unc', data=lgh.Lnu_unc)
            if (lgh.sfh.type == 'piecewise'):
                properties.create_dataset('steps_bounds', data=lgh.ages)

            properties.create_dataset('mstar/lo', data=mstar_q[0])
            properties.create_dataset('mstar/med', data=mstar_q[1])
            properties.create_dataset('mstar/hi', data=mstar_q[2])
            properties.create_dataset('mstar/best', data=mstar[bestfit])

            pvalue,_,_ = ppc(lgh, samples, logprob_samples)

            properties.create_dataset('pvalue', data=pvalue)

def postprocess_catalog_mle(res_filenames,
                            model_filenames,
                            model_mode='json',
                            names=None,
                            catalog_name='postprocessed_catalog.hdf5'):
    assert (len(res_filenames) == len(model_filenames)), "We require the same number of result and model filenames."
    if names is not None:
        assert (len(names) == len(res_filenames)), "We require the same number of source names as results."
    else:
        # Assume that anything before the first underscore in the chain files is the name
        names = [path.splitext(path.basename(s))[0].split('_')[0] for s in res_filenames]

    assert (model_mode in ['json', 'pickle']), "'model_mode' must be either 'json' or 'pickle'."

    with h5py.File(catalog_name, 'w') as outfile:

        for i, cf, mf, n in tqdm(zip(np.arange(len(res_filenames)), res_filenames, model_filenames, names), total=len(res_filenames)):

            with h5py.File(cf, 'r') as f:

                # Writing to in-memory arrays so that samples and logprob_samples don't go
                # out of scope when the file closes. This is probably silly, and contrary to
                # the whole point of HDF5. Could just bump everything under the above context
                # instead.
                bestfit = np.zeros(f['res/bestfit'].shape)
                f['res/bestfit'].read_direct(bestfit)
                chi2_best = np.zeros(f['res/chi2_best'].shape)
                f['res/chi2_best'].read_direct(chi2_best)

            source = outfile.create_group(n)

            source.create_dataset('res/bestfit', data=bestfit)
            source.create_dataset('res/chi2_best', data=chi2_best)

            if (model_mode == 'json'):
                lgh = Lightning.from_json(mf)
            elif (model_mode == 'pickle'):
                with open(mf, 'rb') as f:
                    lgh = pickle.load(f)

            ##### PARAMETERS
            sfh_params, atten_params, dust_params, agn_params, st_xray_params, agn_xray_params, xray_abs_params = lgh._separate_params(bestfit)
            parameters = source.create_group('parameters')
            if (sfh_params is not None):
                for j,pname in enumerate(lgh.sfh.param_names):
                    parameters.create_dataset('sfh/%s/best' % (pname), data=sfh_params)
            if (atten_params is not None):
                for j,pname in enumerate(lgh.atten.param_names):
                    parameters.create_dataset('atten/%s/best' % (pname), data=atten_params)
            if (dust_params is not None):
                for j,pname in enumerate(lgh.dust.param_names):
                    parameters.create_dataset('dust/%s/best' % (pname), data=dust_params)
            if (agn_params is not None):
                for j,pname in enumerate(lgh.agn.param_names):
                    parameters.create_dataset('agn/%s/best' % (pname), data=agn_params)
            if (st_xray_params is not None):
                for j,pname in enumerate(lgh.xray_stellar_em.param_names):
                    parameters.create_dataset('xray_stellar/%s/best' % (pname), data=st_xray_params)
            if (agn_xray_params is not None):
                for j,pname in enumerate(lgh.xray_agn_em.param_names):
                    parameters.create_dataset('xray_agn/%s/best' % (pname), data=agn_xray_params)
            if (xray_abs_params is not None):
                for j,pname in enumerate(lgh.xray_abs_intr.param_names):
                    parameters.create_dataset('xray_abs/%s/best' % (pname), data=xray_abs_params)

            ##### PROPERTIES
            properties = source.create_group('properties')
            if (lgh.sfh.type == 'piecewise'):
                mstar = lgh.sfh.sum(sfh_params, lgh.stars.mstar)
            else:
                mstar = lgh.sfh.integrate(sfh_params, lgh.stars.mstar)

            #mstar_q = np.nanquantile(mstar, q=(0.16, 0.50, 0.84))

            properties.create_dataset('redshift', data=lgh.redshift)
            properties.create_dataset('lumdist', data=lgh.DL)
            properties.create_dataset('filter_labels', data=lgh.filter_labels)
            properties.create_dataset('lnu', data=lgh.Lnu_obs)
            properties.create_dataset('lnu_unc', data=lgh.Lnu_unc)
            if (lgh.sfh.type == 'piecewise'):
                properties.create_dataset('steps_bounds', data=lgh.ages)

            properties.create_dataset('mstar/best', data=mstar)

            #pvalue,_,_ = ppc(lgh, samples, logprob_samples)

            #properties.create_dataset('pvalue', data=pvalue)


def postprocess_catalog(res_filenames,
                        model_filenames,
                        solver_mode='mcmc',
                        model_mode='json',
                        names=None,
                        catalog_name='postprocessed_catalog.hdf5'):
    '''Given lists of chain files and model files, merge the results into a postprocessed catalog.

    This postprocessing script is for *samplers*, not for maximum likelihood methods. Luckily
    I haven't fully implemented any of the maximum likelihood methods yet.

    This script uses h5py to produce an output file in HDF5 format. I've made this choice to allow for
    non-homogeneous model setups, e.g. different numbers of bandpasses and parameters per source.
    The structure and content of the HDF5 file is as follows (for each source):

    └──sourcename
        ├── mcmc
        │   ├── logprob_samples (Nsamples)
        │   └── samples (Nsamples, Nparams)
        ├── parameters
        │   ├── modelname
        │   │   └── parametername
        │   │       ├── best ()
        │   │       ├── hi ()
        │   │       ├── lo ()
        │   │       └── med ()
        └── properties
            ├── filter_labels (Nfilters)
            ├── lnu (Nfilters)
            ├── lnu_unc (Nfilters)
            ├── lumdist ()
            ├── mstar
            │   ├── best ()
            │   ├── hi ()
            │   ├── lo ()
            │   └── med ()
            ├── redshift ()
            └── pvalue ()

    The "modelname" and "parametername" groups under the "parameters" group repeat for every model
    and parameter. For piecewise-constant SFHs the "properties" group also contains the age bin edges
    for the SFH. Quantiles ("*/lo" and "*/hi") are computed at the 16 and 84th percentile.

    For solver_mode='mcmc', the chains are also expected to be in HDF5 format, with the following structure:

    mcmc
    ├── logprob_samples (Nsamples)
    ├── samples (Nsamples, Nparams)
    └── autocorr (Nparams)

    Whereas for solver_mode='mle', the results are expected to be formated as:

    res
    ├── bestfit (Nparams)
    └── chi2_best ()


    I'll provide a function to make such HDF5 result files soon if I haven't already.


    Parameters
    ----------
    res_filenames : array-like, str
        A list of filenames pointing to the result files.
    model_filenames : array-like, str
        A list of filenames poitning to the model files (either json or pickles).
    model_mode : str
        Method for model serialization, either "json" or "pickle". (Default: "json")
    names : array-like, str
        Names for each of the galaxies. If None (default), we'll just guess
        based on the filenames of the chains assuming that they're named something like
        [NAME]_chain.npy.
    catalog_name : str
        Default: "postprocessed_catalog.hdf5"

    Returns
    -------
    Nothing.

    Notes
    -----
    TODO:
    - Allow user-supplied quantiles.
    - Add additional properties; allow user specification of properties? That might be a whole
      chore.

    '''

    if (solver_mode == 'mcmc'):
        postprocess_catalog_mcmc(res_filenames,
                                 model_filenames,
                                 model_mode=model_mode,
                                 names=names,
                                 catalog_name=catalog_name)
    elif (solver_mode == 'mle'):
        postprocess_catalog_mle(res_filenames,
                                model_filenames,
                                model_mode=model_mode,
                                names=names,
                                catalog_name=catalog_name)
    else:
        print('solver_mode must be one of "mcmc" or "mle".')
        return None


if (__name__ == '__main__'):

    import sys
    # I'll add argparse to this at some point maybe.
    # Then part of the install will be adding something like
    # alias "lightning_postprocess=python $lightning_dir/lightning/postprocessing.py"
    # to .bash_profile so that you can run it as a CLI app

    res_listfile = sys.argv[1]
    model_listfile = sys.argv[2]
    outname = sys.arv[3]
    mode = sys.argv[4]

    postprocess_catalog(res_listfile,
                        model_listfile,
                        catalog_name=outname,
                        solver_mode=mode)
