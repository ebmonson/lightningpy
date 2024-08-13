import numpy as np
from scipy.interpolate import interp1d
import matplotlib as mpl
import matplotlib.pyplot as plt

from .plots import ModelBand

def ppc(lgh, samples, logprob_samples, Nrep=1000, seed=None, counts_dist='gaussian'):
    '''Compute the posterior predictive check p-value given a set of samples and a Lightning object

    Parameters
    ----------
    lgh : lightning.Lightning object
        Used to compute chi2 values.
    samples : np.ndarray, (Nsamples, Nparam), float
        The samples to compute the PPC with. Note that
        this must also include any constant parameters,
        since we need them to compute the model luminosities.
    logprob_samples : np.ndarray, (Nsamples,), float
        The logprob corresponding to each sample. Used to
        weight samples.
    Nrep : int
        Number of realizations of the data to compute from
        the model.
    seed : float
        Seed for random number generation.

    Returns
    -------
    p-value : float
        A single p-value for the given chain. Extremely low p-values
        indicate underfitting, that the model is not flexible enough to
        produce the observed variation in the data (or that the uncertainties
        on the data are too small), while extremely high p-values (close to 1)
        may indicate over-fitting, where the model is *too* flexible (or the
        uncertainties overly conservative).

    Notes
    -----
    The implementation of PPC here is ported from IDL Lightning.

    '''

    rng = np.random.default_rng(seed)

    # We construct the CDF in order to draw
    # weighted samples from the chains
    sort = np.argsort(np.exp(logprob_samples))
    pdf = np.exp(logprob_samples)[sort]
    cdf = np.cumsum(pdf)
    cdf /= np.amax(cdf)

    # Check if all elements of the CDF are finite?
    # this was a concern due to I/O choices we made
    # in IDL lightning postproc, maybe not here.

    # Invert the CDF to get the index.
    finterp = interp1d(cdf, np.arange(len(cdf)), kind='nearest')
    idcs = finterp(rng.uniform(size=Nrep))
    idcs = idcs.astype(int)

    # For each set of sample parameters, the model luminosity is computed
    # How do we account for the constant parameters? Just make sure that the
    # sample array that we're given here already includes them? That's kind of a
    # hassle.
    Lmod,_ = lgh.get_model_lnu(samples[sort,:][idcs,:])

    # and perturbed by the uncertainties on the data
    total_unc2 = lgh.Lnu_unc[None,:]**2 + (lgh.model_unc * Lmod)**2
    Lmod_perturbed = rng.normal(loc=Lmod, scale=np.sqrt(total_unc2))

    # Lmod_perturbed are now fake "observed" data, assuming that the model is correct.
    # We calculate chi2 comparing these new "observations" with the original observations
    # and with the unperturbed Lmod.
    chi2_obs = np.nansum((lgh.Lnu_obs[None,:] - Lmod_perturbed)**2 / total_unc2, axis=-1)
    chi2_rep = np.nansum((Lmod - Lmod_perturbed)**2 / total_unc2, axis=-1)

    # Now we calculate the X-ray contribution to both kinds of chi2, if applicable.
    if ((lgh.xray_stellar_em is not None) or (lgh.xray_agn_em is not None)):
        xray_mask = np.array(['XRAY' in s for s in lgh.filter_labels])
        if (lgh.xray_mode == 'counts'):
            net_counts = lgh.xray_counts
            net_counts_unc = lgh.xray_counts_unc
            counts_mod = lgh.get_xray_model_counts(samples[sort,:][idcs,:])
            counts_total_unc2 = lgh.xray_counts_unc[None,:]**2 + (lgh.model_unc * counts_mod)**2
            # We have the option here, of perturbing our model counts by assuming that they're
            # Poisson distributed, or by assuming that they follow a Gaussian distribution with the
            # same uncertainty we've previously assumed for the data.
            if (counts_dist.lower() == 'poisson'):
                counts_perturbed = rng.poisson(lam=counts_mod)
            elif (counts_dist.lower() == 'gaussian'):
                counts_perturbed = rng.normal(loc=counts_mod, scale=np.sqrt(counts_total_unc2))
            else:
                raise ValueError("'counts_dist' must be either 'poisson' or 'gaussian'.")

            xray_chi2_obs = np.nansum((net_counts[None,xray_mask] - counts_perturbed[:,xray_mask]) ** 2 / counts_total_unc2[:,xray_mask], axis=-1)
            xray_chi2_rep = np.nansum((counts_mod[:,xray_mask] - counts_perturbed[:,xray_mask]) ** 2 / counts_total_unc2[:,xray_mask], axis=-1)

        else:
            Lmod_xray,_ = lgh.get_xray_model_lnu(samples[sort,:][idcs,:])
            xray_total_unc2 = lgh.Lnu_unc[None,:]**2 + (lgh.model_unc * Lmod_xray)**2

            Lmod_xray_perturbed = rng.normal(loc=Lmod_xray, scale=np.sqrt(xray_total_unc2))

            xray_chi2_obs = np.nansum((lgh.Lnu_obs[None,xray_mask] - Lmod_xray_perturbed[:,xray_mask])**2 / xray_total_unc2[:,xray_mask], axis=-1)
            xray_chi2_rep = np.nansum((Lmod_xray[:,xray_mask] - Lmod_xray_perturbed[:,xray_mask])**2 / xray_total_unc2[:,xray_mask], axis=-1)

        chi2_obs += xray_chi2_obs
        chi2_rep += xray_chi2_rep

    # The p-value is then the fraction of samples with new chi2 greater than the
    # old chi2
    p_value = np.count_nonzero(chi2_rep > chi2_obs) / float(Nrep)

    return p_value, chi2_rep, chi2_obs

def ppc_sed(lgh, samples, logprob_samples, Nrep=1000, seed=None, ax=None, normalize=False, counts_dist='gaussian'):
    '''Make an SED plot representing the posterior predictive check.

    The idea is that this can serve as a diagnostic plot showing where the model may
    be too inflexible to reproduce individual datapoints. This may show you if, e.g.
    your low p-value is driven by an individual band.

    Parameters
    ----------
    lgh : lightning.Lightning object
        Used to compute chi2 values.
    samples : np.ndarray, (Nsamples, Nparam), float
        The samples to compute the PPC with. Note that
        this must also include any constant parameters,
        since we need them to compute the model luminosities.
    logprob_samples : np.ndarray, (Nsamples,), float
        The logprob corresponding to each sample. Used to
        weight samples.
    Nrep : int
        Number of realizations of the data to compute from
        the model.
    seed : float
        Seed for random number generation. Useful for matching your
        PPC p-value to the plot.
    ax : matplotlib.axis.Axes
        Axes to draw the plot into.
    normalize : bool
        If True, the data and quantiles of the reproduced data are divided
        by the median of the reproduced data before plotting.

    Returns
    -------
    PPC SED plot figure: an SED plot showing the quantile bands of the reproduced
    data, with the observed data overplotted.

    '''

    rng = np.random.default_rng(seed)

    # We construct the CDF in order to draw
    # weighted samples from the chains
    sort = np.argsort(np.exp(logprob_samples))
    pdf = np.exp(logprob_samples)[sort]
    cdf = np.cumsum(pdf)
    cdf /= np.amax(cdf)

    # We need these for the case where
    # the X-ray model was fit with counts
    bestfit_samples = samples[sort,:][-1,:]

    # Invert the CDF to get the index.
    finterp = interp1d(cdf, np.arange(len(cdf)), kind='nearest')
    idcs = finterp(rng.uniform(size=Nrep))
    idcs = idcs.astype(int)

    # For each set of sample parameters, the model luminosity is computed
    # How do we account for the constant parameters? Just make sure that the
    # sample array that we're given here already includes them? That's kind of a
    # hassle.
    Lmod,_ = lgh.get_model_lnu(samples[sort,:][idcs,:])

    # and perturbed by the uncertainties on the data
    total_unc2 = lgh.Lnu_unc[None,:]**2 + (lgh.model_unc * Lmod)**2
    Lmod_perturbed = rng.normal(loc=Lmod, scale=np.sqrt(total_unc2))

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.gcf()

    med = np.median(lgh.nu_obs[None,:] * Lmod_perturbed, axis=0)

    if (normalize):
        norm = med
        unit_str = r'a.u.'
    else:
        norm = np.ones_like(med)
        unit_str = r'$\rm L_{\odot}$'

    # The filters might not be in ascending order of
    # wavelength, so we sort them here.
    wave_idcs = np.argsort(lgh.wave_obs)
    band = ModelBand(lgh.wave_obs[wave_idcs])
    for mod in Lmod_perturbed:
        band.add(lgh.nu_obs[wave_idcs] * mod[wave_idcs] / norm[wave_idcs])

    band.shade(q=(0.005, 0.995), color='darkorange', alpha=0.2, label='99%', ax=ax)
    band.shade(q=(0.025, 0.975), color='darkorange', alpha=0.3, label='95%', ax=ax)
    band.shade(q=(0.160, 0.840), color='darkorange', alpha=0.4, label='68%', ax=ax)
    band.line(q=0.5, color='darkorange', label='Median', ax=ax)

    ax.scatter(lgh.wave_obs,
               lgh.nu_obs * lgh.Lnu_obs / norm,
               marker='x',
               color='k',
               zorder=10,
               label='Data')

    # Now we calculate the X-ray contribution to both kinds of chi2, if applicable.
    if ((lgh.xray_stellar_em is not None) or (lgh.xray_agn_em is not None)):
        xray_mask = np.array(['XRAY' in s for s in lgh.filter_labels])

        if (lgh.xray_stellar_em is not None):
            xray_wave_obs = lgh.xray_stellar_em.wave_obs
            xray_nu_obs = lgh.xray_stellar_em.nu_obs
        else:
            xray_wave_obs = lgh.xray_agn_em.wave_obs
            xray_nu_obs = lgh.xray_agn_em.nu_obs

        if (lgh.xray_mode == 'counts'):
            # net_counts = lgh.xray_counts
            # net_counts_unc = lgh.xray_counts_unc
            counts_best = lgh.get_xray_model_counts(bestfit_samples)
            lnu_xray_best,_ = lgh.get_xray_model_lnu(bestfit_samples)
            lnu_obs_xray = lgh.xray_counts[xray_mask] / counts_best[xray_mask]  * lnu_xray_best[xray_mask]

            Lmod_xray,_ = lgh.get_xray_model_lnu(samples[sort,:][idcs,:])
            counts_mod = lgh.get_xray_model_counts(samples[sort,:][idcs,:])
            counts_total_unc2 = lgh.xray_counts_unc[None,:]**2 + (lgh.model_unc * counts_mod)**2
            # We have the option here, of perturbing our model counts by assuming that they're
            # Poisson distributed, or by assuming that they follow a Gaussian distribution with the
            # same uncertainty we've previously assumed for the data.
            if (counts_dist.lower() == 'poisson'):
                counts_perturbed = rng.poisson(lam=counts_mod)
            elif (counts_dist.lower() == 'gaussian'):
                counts_perturbed = rng.normal(loc=counts_mod, scale=np.sqrt(counts_total_unc2))
            else:
                raise ValueError("'counts_dist' must be either 'poisson' or 'gaussian'.")

            Lmod_xray_perturbed = lgh.xray_counts[None,xray_mask] / counts_perturbed[:,xray_mask] * Lmod_xray[:,xray_mask]

        else:
            lnu_obs_xray = lgh.Lnu_obs[xray_mask]

            Lmod_xray,_ = lgh.get_xray_model_lnu(samples[sort,:][idcs,:])
            xray_total_unc2 = lgh.Lnu_unc[None,:]**2 + (lgh.model_unc * Lmod_xray)**2

            Lmod_xray_perturbed = rng.normal(loc=Lmod_xray, scale=np.sqrt(xray_total_unc2))[:,xray_mask]

        xray_med = np.median(xray_nu_obs[xray_mask] * Lmod_xray_perturbed, axis=0)

        xray_wave_idcs = np.argsort(xray_wave_obs[xray_mask])
        xray_band = ModelBand(xray_wave_obs[xray_mask][xray_wave_idcs])
        if (normalize):
            xray_norm = xray_med
        else:
            xray_norm = np.ones_like(xray_med)

        for mod in Lmod_xray_perturbed:
            xray_band.add(xray_nu_obs[xray_mask][xray_wave_idcs] * mod[xray_wave_idcs] / xray_norm[xray_wave_idcs])

        xray_band.shade(q=(0.005, 0.995), color='darkorange', alpha=0.2, ax=ax)
        xray_band.shade(q=(0.025, 0.975), color='darkorange', alpha=0.3, ax=ax)
        xray_band.shade(q=(0.160, 0.840), color='darkorange', alpha=0.4, ax=ax)
        xray_band.line(q=0.5, color='darkorange', ax=ax)

        ax.scatter(xray_wave_obs[xray_mask],
                   xray_nu_obs[xray_mask] * lnu_obs_xray / xray_norm,
                   marker='x',
                   color='k',
                   zorder=10)

    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_xlabel(r'Observed-Frame Wavelength [$\rm \mu m$]')
    ax.set_ylabel(r'$\nu L_{\nu}$ [%s]' % unit_str)

    ax.legend(loc='lower left')

    return fig, ax
