import numpy as np
from scipy.interpolate import interp1d
import matplotlib as mpl
import matplotlib.pyplot as plt

def ppc(lgh, samples, logprob_samples, Nrep=1000, seed=None):
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

    Outputs
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


    # Note that we have not yet calculate the X-ray contribution to chi2

    # The p-value is then the fraction of samples with new chi2 greater than the
    # old chi2
    p_value = np.count_nonzero(chi2_rep > chi2_obs) / float(Nrep)

    return p_value, chi2_rep, chi2_obs

def ppc_sed(lgh, samples, logprob_samples, Nrep=1000, seed=None, ax=None, normalize=False):
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

    Outputs
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

    # Note that the X-rays are not included yet.

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.gcf()

    # ax.violinplot(lgh.wave_obs,
    #               (lgh.nu_obs[None,:] * Lmod_perturbed).T)

    qq = np.quantile(lgh.nu_obs[None,:] * Lmod_perturbed,
                     q=(0.005, 0.025, 0.16, 0.50, 0.84, 0.975, 0.995),
                     axis=0)

    if (normalize):
        norm = qq[3]
        unit_str = r'a.u.'
    else:
        norm = np.ones_like(qq[3])
        unit_str = r'$\rm L_{\odot}$'

    ax.fill_between(lgh.wave_obs,
                    qq[0] / norm, qq[-1] / norm,
                    color='darkorange',
                    alpha=0.2,
                    label='99%')
    ax.fill_between(lgh.wave_obs,
                    qq[1] / norm, qq[-2] / norm,
                    color='darkorange',
                    alpha=0.3,
                    label='95%')
    ax.fill_between(lgh.wave_obs,
                    qq[2] / norm, qq[-3] / norm,
                    color='darkorange',
                    alpha=0.4,
                    label='68%')
    ax.plot(lgh.wave_obs,
            qq[3] / norm,
            color='darkorange',
            label='Median')

    ax.scatter(lgh.wave_obs,
               lgh.nu_obs * lgh.Lnu_obs / norm,
               marker='x',
               color='k',
               zorder=10,
               label='Data')

    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_xlabel(r'Observed-Frame Wavelength [$\rm \mu m$]')
    ax.set_ylabel(r'$\nu L_{\nu}$ [%s]' % unit_str)

    ax.legend(loc='lower left')

    return fig, ax
