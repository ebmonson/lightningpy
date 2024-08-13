import numpy as np
import matplotlib.pyplot as plt
from .band import ModelBand

def sed_plot_bestfit(lgh, samples, logprob_samples,
                     plot_components=False,
                     plot_unatt=False,
                     ax=None,
                     xlim=(0.1, 1200), ylim=(0.9*1e7,None),
                     xlabel=r'Observed-Frame Wavelength $[\rm \mu m]$',
                     ylabel=r'$\nu L_{\nu}\ [\rm L_\odot]$',
                     stellar_unatt_kwargs={'color':'dodgerblue', 'label':'Unattenuated stellar pop.'},
                     stellar_att_kwargs={'color':'red', 'label':'Attenuated stellar pop.'},
                     agn_kwargs={'color':'darkorange', 'label':'AGN'},
                     dust_kwargs={'color':'green', 'label':'Dust'},
                     total_kwargs={'color':'slategray', 'label': 'Total model'},
                     data_kwargs={'marker':'D', 'color':'k', 'markerfacecolor':'k', 'capsize':2, 'linestyle':'', 'label': 'Data'},
                     show_legend=True,
                     legend_kwargs={'loc':'upper right', 'frameon':False},
                     uplim_sigma=3,
                     uplim_kwargs={'marker':r'$\downarrow$', 'color':'k'}
                     ):
    '''Best-fit SED plot.

    Selects the highest log-probability model from the chain. Individual
    components may be plotted.

    Parameters
    ----------
    lgh : lightning.Lightning object
        Used to assign names (and maybe units) to the sample
        dimensions.
    samples : np.ndarray, (Nsamples, Nparam), float
        The sampled parameters. Note that this should also include
        any constant parameters.
    logprob_samples : np.ndarray, (Nsamples,), float
        Log-probability chain.
    plot_components : bool
        If ``True``, plot the components of the SED.
    plot_unatt : bool
        If ``True``, also plot the unattenuated stellar SED.
    ax : matplotlib.axes.Axes
        Axes to draw the plot in. (Default: None)
    xlim : tuple
    ylim : tuple
    xlabel : str
    ylabel : str
    stellar_unatt_kwargs : dict
    stellar_att_kwargs : dict
    agn_kwargs : dict
    dust_kwargs : dict
    total_kwargs : dict
    data_kwargs : dict
    show_legend : bool
    legend_kwargs : dict
    uplim_sigma : int
        How many sigma should upper limits be drawn at? (Default: 3)
    uplim_kwargs : dict
        Each of the above ``*_kwargs`` parameters is a dict containing keyword arguments describing the color, style,
        label, etc. of the corresponding plot element, passed through to the appropriate ``matplotlib`` function.

    Returns
    -------
    fig : Matplotlib figure containing the plot
    ax : Axes containing the plot

    '''

    bestfit = np.argmax(logprob_samples)
    bestfit_samples = samples[bestfit,:]

    lnu_hires_att_best, lnu_hires_unatt_best = lgh.get_model_lnu_hires(bestfit_samples)
    lnu_att_best, lnu_unatt_best = lgh.get_model_lnu(bestfit_samples)

    lnu_unc_total = np.sqrt(lgh.Lnu_unc**2 + (lgh.model_unc * lnu_att_best)**2)

    uplim_mask = (lgh.Lnu_obs == 0) & (lgh.Lnu_unc > 0)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6,5))
    else:
        fig = ax.get_figure()

    # We assemble the legend manually to avoid
    # duplicating entries when the X-ray model
    # is plotted.
    legend_elements = []

    if plot_components:
        # Oh boy that's a mouthful of a function name
        lnu_components = lgh.get_model_components_lnu_hires(bestfit_samples)

        if (plot_unatt):
            l, = ax.plot(lgh.wave_grid_obs, lgh.nu_grid_obs * lnu_components['stellar_unattenuated'], **stellar_unatt_kwargs)
            legend_elements.append(l)

        l, = ax.plot(lgh.wave_grid_obs, lgh.nu_grid_obs * lnu_components['stellar_attenuated'], **stellar_att_kwargs)
        legend_elements.append(l)

        if lgh.agn is not None:
            l, = ax.plot(lgh.wave_grid_obs, lgh.nu_grid_obs * lnu_components['agn'], **agn_kwargs)
            legend_elements.append(l)

        if lgh.dust is not None:
            l, = ax.plot(lgh.wave_grid_obs, lgh.nu_grid_obs * lnu_components['dust'], **dust_kwargs)
            legend_elements.append(l)

    if ((lgh.xray_stellar_em is not None) or (lgh.xray_agn_em is not None)):

        if plot_components:

            if (lgh.xray_stellar_em is not None):
                if (plot_unatt):
                    ax.plot(lgh.xray_wave_grid_obs, lgh.xray_nu_grid_obs * lnu_components['xray_stellar_unabsorbed'], **stellar_unatt_kwargs)

                ax.plot(lgh.xray_wave_grid_obs, lgh.xray_nu_grid_obs * lnu_components['xray_stellar_absorbed'], **stellar_att_kwargs)

            if (lgh.xray_agn_em is not None):
                ax.plot(lgh.xray_wave_grid_obs, lgh.xray_nu_grid_obs * lnu_components['xray_agn_absorbed'], **agn_kwargs)


        lnu_xray_hires_abs_best, lnu_xray_hires_unabs_best = lgh.get_xray_model_lnu_hires(bestfit_samples)

        ax.plot(lgh.xray_wave_grid_obs, lgh.xray_nu_grid_obs * lnu_xray_hires_abs_best,
                **total_kwargs)


        # Establish the observed wavelengths/frequencies
        xray_mask = np.array(['XRAY' in s for s in lgh.filter_labels])
        if (lgh.xray_stellar_em is not None):
            xray_wave_obs = lgh.xray_stellar_em.wave_obs
            xray_nu_obs = lgh.xray_stellar_em.nu_obs
        else:
            xray_wave_obs = lgh.xray_agn_em.wave_obs
            xray_nu_obs = lgh.xray_agn_em.nu_obs

        # If we fit the count spectrum we need
        # to estimate luminosities for the plot.
        if (lgh.xray_mode == 'counts'):

            counts_xray_best = lgh.get_xray_model_counts(bestfit_samples)
            lnu_xray_best,_ = lgh.get_xray_model_lnu(bestfit_samples)

            lnu_obs_xray_best = lgh.xray_counts[xray_mask] / counts_xray_best[xray_mask]  * lnu_xray_best[xray_mask]
            lnu_unc_xray_best = lnu_obs_xray_best / (lgh.xray_counts[xray_mask] / lgh.xray_counts_unc[xray_mask])

            ax.errorbar(xray_wave_obs[xray_mask],
                        xray_nu_obs[xray_mask] * lnu_obs_xray_best,
                        yerr=xray_nu_obs[xray_mask] * lnu_unc_xray_best,
                        **data_kwargs)
        else:

            ax.errorbar(xray_wave_obs[~uplim_mask & xray_mask],
                        xray_nu_obs[~uplim_mask & xray_mask] * lgh.Lnu_obs[~uplim_mask & xray_mask],
                        yerr=xray_nu_obs[~uplim_mask & xray_mask] * lgh.Lnu_unc[~uplim_mask & xray_mask],
                        **data_kwargs)

            ax.scatter(xray_wave_obs[uplim_mask & xray_mask],
                       uplim_sigma * xray_nu_obs[uplim_mask & xray_mask] * lgh.Lnu_unc[uplim_mask & xray_mask],
                       **uplim_kwargs)


    l, = ax.plot(lgh.wave_grid_obs, lgh.nu_grid_obs * lnu_hires_att_best,
                **total_kwargs)
    legend_elements.append(l)

    l = ax.errorbar(lgh.wave_obs[~uplim_mask],
                    lgh.nu_obs[~uplim_mask] * lgh.Lnu_obs[~uplim_mask],
                    yerr=lgh.nu_obs[~uplim_mask] * lnu_unc_total[~uplim_mask],
                    **data_kwargs)
    legend_elements.append(l)

    ax.scatter(lgh.wave_obs[uplim_mask],
               uplim_sigma * lgh.nu_obs[uplim_mask] * lgh.Lnu_unc[uplim_mask],
               **uplim_kwargs)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    if show_legend:
        ax.legend(handles=legend_elements, **legend_kwargs)

    return fig, ax

def sed_plot_delchi(lgh, samples, logprob_samples, ax=None,
                    xlim=(0.1, 1200), ylim=(-5.1, 5.1),
                    lines=[0,-1,1], linecolors=['slategray'], linestyles=['-','--','--'],
                    xlabel=r'Observed-Frame Wavelength $[\rm \mu m]$',
                    ylabel=r'Residual $[\sigma]$',
                    data_kwargs={'marker':'D', 'color':'k', 'markerfacecolor':'k', 'capsize':2, 'linestyle':'', 'label': 'Data'},
                    uplim_sigma=3,
                    uplim_kwargs={'marker':r'$\downarrow$', 'color':'k'}
                    ):
    r'''Delta-chi residuals for the best-fit SED.

    .. math::
        \delta \chi = (L^{\rm obs}_\nu - L^{\rm mod}_\nu) / \sigma


    Selects the highest log-probability model from the chain.

    Parameters
    ----------
    lgh : lightning.Lightning object
        Used to assign names (and maybe units) to the sample
        dimensions.
    samples : np.ndarray, (Nsamples, Nparam), float
        The sampled parameters. Note that this should also include
        any constant parameters.
    logprob_samples : np.ndarray, (Nsamples,), float
        Log-probability chain.
    xlim : tuple
    ylim : tuple
    lines : list or tuple
        Values to draw horizontal guide lines at (in sigma). (Default: [-1,0,1])
    linecolors : list or str
        Corresponding colors for the guide lines. (Default: 'slategray')
    linestyles : list or str
        Corresponding styles for the guide lines. (Default: ['-', '--', '--'])
    xlabel : str
    ylabel : str
    data_kwargs : dict
    uplim_sigma : int
        How many sigma should upper limits be drawn at? (Default: 3)
    uplim_kwargs : dict
        Each of the above ``*_kwargs`` parameters is a dict containing keyword arguments describing the color, style,
        label, etc. of the corresponding plot element, passed through to the appropriate ``matplotlib`` function.

    Returns
    -------
    fig : Matplotlib figure containing the plot
    ax : Axes containing the plot

    '''

    bestfit = np.argmax(logprob_samples)
    bestfit_samples = samples[bestfit,:]

    lnu_att_best, lnu_unatt_best = lgh.get_model_lnu(bestfit_samples)

    lnu_unc_total = np.sqrt(lgh.Lnu_unc**2 + (lgh.model_unc * lnu_att_best)**2)

    uplim_mask = (lgh.Lnu_obs == 0) & (lgh.Lnu_unc > 0)
    Nuplims = np.count_nonzero(uplim_mask)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6,5))
    else:
        fig = ax.get_figure()

    delchi = (lgh.Lnu_obs - lnu_att_best) / lnu_unc_total

    if lines is not None:
        Nlines = int(len(lines))
        if isinstance(linecolors, str): linecolors = [linecolors]
        if len(linecolors) == 1: linecolors = Nlines * linecolors
        if isinstance(linestyles, str): linestyles = [linestyles]
        if len(linestyles) == 1: linestyles = Nlines * linestyles

        for val, c, s in zip(lines, linecolors, linestyles):
            ax.axhline(val, color=c, linestyle=s)

    ax.errorbar(lgh.wave_obs[~uplim_mask],
                delchi[~uplim_mask],
                yerr=(1 + np.zeros_like(delchi[~uplim_mask])),
                **data_kwargs)

    if (Nuplims > 0):
        delchi_uplim = (uplim_sigma * lgh.Lnu_unc - lnu_att_best) / lnu_unc_total
        ax.scatter(lgh.wave_obs[uplim_mask],
                   delchi_uplim[uplim_mask],
                   **uplim_kwargs)

    if ((lgh.xray_stellar_em is not None) or (lgh.xray_agn_em is not None)):

        # Establish the observed wavelengths
        xray_mask = np.array(['XRAY' in s for s in lgh.filter_labels])
        if (lgh.xray_stellar_em is not None):
            xray_wave_obs = lgh.xray_stellar_em.wave_obs
        else:
            xray_wave_obs = lgh.xray_agn_em.wave_obs

        if (lgh.xray_mode == 'counts'):

            counts_xray_best = lgh.get_xray_model_counts(bestfit_samples)
            counts_unc_total = np.sqrt(lgh.xray_counts_unc**2 + (lgh.model_unc * counts_xray_best)**2)

            delchi_xray = (lgh.xray_counts - counts_xray_best) / counts_unc_total

        else:

            lnu_xray_best,_ = lgh.get_xray_model_lnu(bestfit_samples)
            lnu_xray_unc_total = np.sqrt(lgh.Lnu_unc**2 + (lgh.model_unc * lnu_xray_best)**2)

            delchi_xray = (lgh.Lnu_obs - lnu_xray_best) / lnu_xray_unc_total

            if (Nuplims > 0):
                delchi_xray_uplim = (uplim_sigma * lgh.Lnu_unc - lnu_xray_best) / lnu_xray_unc_total
                ax.scatter(lgh.wave_obs[uplim_mask],
                           delchi_xray_uplim[uplim_mask],
                           **uplim_kwargs)

        ax.errorbar(xray_wave_obs[xray_mask],
                    delchi_xray[xray_mask],
                    yerr=(1 + np.zeros_like(delchi_xray[xray_mask])),
                    **data_kwargs)


    ax.set_xscale('log')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    return fig, ax

def sed_plot_morebayesian(lgh, samples,
                          logprob_samples,
                          plot_components=False,
                          plot_unatt=False,
                          ax=None,
                          xlim=(0.1, 1200), ylim=(0.9*1e7,None),
                          xlabel=r'Observed-Frame Wavelength $[\rm \mu m]$',
                          ylabel=r'$\nu L_{\nu}\ [\rm L_\odot]$',
                          stellar_unatt_kwargs={'color':'dodgerblue', 'label':'Unattenuated stellar pop.', 'alpha':0.5},
                          stellar_att_kwargs={'color':'red', 'label':'Attenuated stellar pop.', 'alpha':0.5},
                          agn_kwargs={'color':'darkorange', 'label':'AGN', 'alpha':0.5},
                          dust_kwargs={'color':'green', 'label':'Dust', 'alpha':0.5},
                          total_kwargs={'color':'slategray', 'label': 'Total model', 'alpha':0.5},
                          data_kwargs={'marker':'D', 'color':'k', 'markerfacecolor':'k', 'capsize':2, 'linestyle':'', 'label': 'Data'},
                          show_legend=True,
                          legend_kwargs={'loc':'upper right', 'frameon':False},
                          uplim_sigma=3,
                          uplim_kwargs={'marker':r'$\downarrow$', 'color':'k'}
                          ):
    '''More bayesian visualization of the fit, agnostic of the best fit. Better name TBD

    Shows the 16th-84th percentile range of the model Lnu. Currently this doesn't
    show the median or best-fit, just shaded polygons indicating the range.

    Parameters
    ----------
    lgh : lightning.Lightning object
        Used to assign names (and maybe units) to the sample
        dimensions.
    samples : np.ndarray, (Nsamples, Nparam), float
        The sampled parameters. Note that this should also include
        any constant parameters.
    logprob_samples : np.ndarray, (Nsamples,), float
        Log-probability chain.
    plot_components : bool
        If ``True``, plot the components of the SED.
    plot_unatt : bool
        If ``True``, also plot the unattenuated stellar SED.
    ax : matplotlib.axes.Axes
        Axes to draw the plot in. (Default: None)
    xlim : tuple
    ylim : tuple
    xlabel : str
    ylabel : str
    stellar_unatt_kwargs : dict
    stellar_att_kwargs : dict
    agn_kwargs : dict
    dust_kwargs : dict
    total_kwargs : dict
    data_kwargs : dict
    show_legend : bool
    legend_kwargs : dict
    uplim_sigma : int
        How many sigma should upper limits be drawn at? (Default: 3)
    uplim_kwargs : dict
        Each of the above ``*_kwargs`` parameters is a dict containing keyword arguments describing the color, style,
        label, etc. of the corresponding plot element, passed through to the appropriate ``matplotlib`` function.

    Returns
    -------
    fig : Matplotlib figure containing the plot
    ax : Axes containing the plot
    '''

    bestfit = np.argmax(logprob_samples)
    bestfit_samples = samples[bestfit,:]

    lnu_hires_att, lnu_hires_unatt = lgh.get_model_lnu_hires(samples)
    lnu_att, lnu_unatt = lgh.get_model_lnu(samples)

    lnu_unc_total = np.median(np.sqrt(lgh.Lnu_unc[None,:]**2 + (lgh.model_unc[None,:] * lnu_att)**2), axis=0)

    uplim_mask = (lgh.Lnu_obs == 0) & (lgh.Lnu_unc > 0)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6,5))
    else:
        fig = ax.get_figure()

    # We assemble the legend manually to avoid
    # duplicating entries when the X-ray model
    # is plotted.
    legend_elements = []

    if plot_components:
        # Oh boy that's a mouthful of a function name
        lnu_components = lgh.get_model_components_lnu_hires(samples)

        if (plot_unatt):
            b = ModelBand(lgh.wave_grid_obs)
            for y in lgh.nu_grid_obs * lnu_components['stellar_unattenuated']: b.add(y)
            l = b.shade(ax=ax, **stellar_unatt_kwargs)
            #l, = b.line(ax=ax, **stellar_unatt_kwargs)
            #l, = ax.plot(lgh.wave_grid_obs, lgh.nu_grid_obs * lnu_components['stellar_unattenuated'], **stellar_unatt_kwargs)
            legend_elements.append(l)

        b = ModelBand(lgh.wave_grid_obs)
        for y in lgh.nu_grid_obs * lnu_components['stellar_attenuated']: b.add(y)
        l = b.shade(ax=ax, **stellar_att_kwargs)
        #l, = b.line(ax=ax, **stellar_att_kwargs)
        #l, = ax.plot(lgh.wave_grid_obs, lgh.nu_grid_obs * lnu_components['stellar_attenuated'], **stellar_att_kwargs)
        legend_elements.append(l)

        if lgh.agn is not None:
            b = ModelBand(lgh.wave_grid_obs)
            for y in lgh.nu_grid_obs * lnu_components['agn']: b.add(y)
            l = b.shade(ax=ax, **agn_kwargs)
            #l, = b.line(ax=ax, **agn_kwargs)
            #l, = ax.plot(lgh.wave_grid_obs, lgh.nu_grid_obs * lnu_components['agn'], **agn_kwargs)
            legend_elements.append(l)

        if lgh.dust is not None:
            b = ModelBand(lgh.wave_grid_obs)
            for y in lgh.nu_grid_obs * lnu_components['dust']: b.add(y)
            l = b.shade(ax=ax, **dust_kwargs)
            #l, = b.line(ax=ax, **dust_kwargs)
            #l, = ax.plot(lgh.wave_grid_obs, lgh.nu_grid_obs * lnu_components['dust'], **dust_kwargs)
            legend_elements.append(l)

    if ((lgh.xray_stellar_em is not None) or (lgh.xray_agn_em is not None)):

        if plot_components:

            if (lgh.xray_stellar_em is not None):
                if (plot_unatt):
                    b = ModelBand(lgh.xray_wave_grid_obs)
                    for y in lgh.xray_nu_grid_obs * lnu_components['xray_stellar_unabsorbed']: b.add(y)
                    b.shade(ax=ax, **stellar_unatt_kwargs)
                    #l, = b.line(ax=ax, **stellar_unatt_kwargs)
                    #ax.plot(lgh.xray_wave_grid_obs, lgh.xray_nu_grid_obs * lnu_components['xray_stellar_unabsorbed'], **stellar_unatt_kwargs)

                b = ModelBand(lgh.xray_wave_grid_obs)
                for y in lgh.xray_nu_grid_obs * lnu_components['xray_stellar_absorbed']: b.add(y)
                b.shade(ax=ax, **stellar_att_kwargs)
                #ax.plot(lgh.xray_wave_grid_obs, lgh.xray_nu_grid_obs * lnu_components['xray_stellar_absorbed'], **stellar_att_kwargs)

            if (lgh.xray_agn_em is not None):
                b = ModelBand(lgh.xray_wave_grid_obs)
                for y in lgh.xray_nu_grid_obs * lnu_components['xray_agn_absorbed']: b.add(y)
                b.shade(ax=ax, **agn_kwargs)
                #ax.plot(lgh.xray_wave_grid_obs, lgh.xray_nu_grid_obs * lnu_components['xray_agn_absorbed'], **agn_kwargs)


        lnu_xray_hires_abs, lnu_xray_hires_unabs = lgh.get_xray_model_lnu_hires(samples)
        b = ModelBand(lgh.xray_wave_grid_obs)
        for y in lgh.xray_nu_grid_obs * lnu_xray_hires_abs: b.add(y)
        b.shade(ax=ax, **total_kwargs)

        # ax.plot(lgh.xray_wave_grid_obs, lgh.xray_nu_grid_obs * lnu_xray_hires_abs_best,
        #         **total_kwargs)


        # Establish the observed wavelengths/frequencies
        xray_mask = np.array(['XRAY' in s for s in lgh.filter_labels])
        if (lgh.xray_stellar_em is not None):
            xray_wave_obs = lgh.xray_stellar_em.wave_obs
            xray_nu_obs = lgh.xray_stellar_em.nu_obs
        else:
            xray_wave_obs = lgh.xray_agn_em.wave_obs
            xray_nu_obs = lgh.xray_agn_em.nu_obs

        # If we fit the count spectrum we need
        # to estimate luminosities for the plot.
        if (lgh.xray_mode == 'counts'):

            # Showing the best-fit conversion between Lnu and counts seems to reflect the
            # actual counts_mod - counts_obs residuals better than any other method I tried.
            counts_xray_best = lgh.get_xray_model_counts(bestfit_samples)
            lnu_xray_best,_ = lgh.get_xray_model_lnu(bestfit_samples)

            lnu_obs_xray_best = lgh.xray_counts[xray_mask] / counts_xray_best[xray_mask]  * lnu_xray_best[xray_mask]
            lnu_unc_xray_best = lnu_obs_xray_best / (lgh.xray_counts[xray_mask] / lgh.xray_counts_unc[xray_mask])

            ax.errorbar(xray_wave_obs[xray_mask],
                        xray_nu_obs[xray_mask] * lnu_obs_xray_best,
                        yerr=xray_nu_obs[xray_mask] * lnu_unc_xray_best,
                        **data_kwargs)

        else:

            ax.errorbar(xray_wave_obs[~uplim_mask & xray_mask],
                        xray_nu_obs[~uplim_mask & xray_mask] * lgh.Lnu_obs[~uplim_mask & xray_mask],
                        yerr=xray_nu_obs[~uplim_mask & xray_mask] * lgh.Lnu_unc[~uplim_mask & xray_mask],
                        **data_kwargs)

            ax.scatter(xray_wave_obs[uplim_mask & xray_mask],
                       uplim_sigma * xray_nu_obs[uplim_mask & xray_mask] * lgh.Lnu_unc[uplim_mask & xray_mask],
                       **uplim_kwargs)


    b = ModelBand(lgh.wave_grid_obs)
    for y in lgh.nu_grid_obs * lnu_hires_att: b.add(y)
    l = b.shade(ax=ax, **total_kwargs)
    # l, = ax.plot(lgh.wave_grid_obs, lgh.nu_grid_obs * lnu_hires_att_best,
    #             **total_kwargs)
    legend_elements.append(l)

    l = ax.errorbar(lgh.wave_obs[~uplim_mask],
                    lgh.nu_obs[~uplim_mask] * lgh.Lnu_obs[~uplim_mask],
                    yerr=lgh.nu_obs[~uplim_mask] * lnu_unc_total[~uplim_mask],
                    **data_kwargs)
    legend_elements.append(l)

    ax.scatter(lgh.wave_obs[uplim_mask],
               uplim_sigma * lgh.nu_obs[uplim_mask] * lgh.Lnu_unc[uplim_mask],
               **uplim_kwargs)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    if show_legend:
        ax.legend(handles=legend_elements, **legend_kwargs)

    return fig, ax

def sed_plot_delchi_morebayesian(lgh, samples, logprob_samples, ax=None,
                                 xlim=(0.1, 1200), ylim=(-5.1, 5.1),
                                 lines=[0,-1,1], linecolors=['slategray'], linestyles=['-','--','--'],
                                 xlabel=r'Observed-Frame Wavelength $[\rm \mu m]$',
                                 ylabel=r'Residual $[\sigma]$',
                                 data_kwargs_inner={'marker':'', 'color':'k', 'markerfacecolor':'k', 'capsize':4, 'linestyle':'', 'alpha':1.0, 'label': 'Data'},
                                 data_kwargs_outer={'marker':'', 'color':'k', 'markerfacecolor':'k', 'capsize':2, 'linestyle':'', 'alpha':0.6},
                                 uplim_sigma=3,
                                 uplim_kwargs={'marker':r'$\downarrow$', 'color':'k'}
                                 ):
    '''More Bayesian delta-chi residuals

    This visualization ignores the best-fit, and instead shows the range of the delta-chi residuals for
    each band.

    I'm not thrilled with the presentation yet. Ideally we would do this with something like a violin plot,
    but in practice there's often enough observations and enough dynamic range in wavelength that the violins overlap
    and squish together. Instead, we plot lines showing the 16th-84th and 5th-95th percentile ranges in delta-chi
    for each band, sort of approximating a violin shape by changing the width of the caps on the lines.

    A better choice might be to sort the observations by wavelength and plot the range of resiudals as a shaded band.

    Parameters
    ----------
    lgh : lightning.Lightning object
        Used to assign names (and maybe units) to the sample
        dimensions.
    samples : np.ndarray, (Nsamples, Nparam), float
        The sampled parameters. Note that this should also include
        any constant parameters.
    logprob_samples : np.ndarray, (Nsamples,), float
        Log-probability chain.
    xlim : tuple
    ylim : tuple
    lines : list or tuple
        Values to draw horizontal guide lines at (in sigma). (Default: [-1,0,1])
    linecolors : list or str
        Corresponding colors for the guide lines. (Default: 'slategray')
    linestyles : list or str
        Corresponding styles for the guide lines. (Default: ['-', '--', '--'])
    xlabel : str
    ylabel : str
    data_kwargs_inner : dict
    data_kwargs_outer : dict
    uplim_sigma : int
        How many sigma should upper limits be drawn at? (Default: 3)
    uplim_kwargs : dict
        Each of the above ``*_kwargs`` parameters is a dict containing keyword arguments describing the color, style,
        label, etc. of the corresponding plot element, passed through to the appropriate ``matplotlib`` function.

    Returns
    -------
    fig : Matplotlib figure containing the plot
    ax : Axes containing the plot

    '''

    lnu_att, lnu_unatt = lgh.get_model_lnu(samples)

    lnu_unc_total = np.sqrt(lgh.Lnu_unc[None,:]**2 + (lgh.model_unc[None,:] * lnu_att)**2)

    uplim_mask = (lgh.Lnu_obs == 0) & (lgh.Lnu_unc > 0)
    Nuplims = np.count_nonzero(uplim_mask)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6,5))
    else:
        fig = ax.get_figure()

    delchi = (lgh.Lnu_obs[None,:] - lnu_att) / lnu_unc_total

    if lines is not None:
        Nlines = int(len(lines))
        if isinstance(linecolors, str): linecolors = [linecolors]
        if len(linecolors) == 1: linecolors = Nlines * linecolors
        if isinstance(linestyles, str): linestyles = [linestyles]
        if len(linestyles) == 1: linestyles = Nlines * linestyles

        for val, c, s in zip(lines, linecolors, linestyles):
            ax.axhline(val, color=c, linestyle=s)

    delchi_mid = np.nanmedian(delchi[:,~uplim_mask], axis=0)
    delchi_16, delchi_84 = np.nanquantile(delchi[:,~uplim_mask], q=(0.16, 0.84), axis=0)
    delchi_05, delchi_95 = np.nanquantile(delchi[:,~uplim_mask], q=(0.05, 0.95), axis=0)
    delchi_01, delchi_99 = np.nanquantile(delchi[:,~uplim_mask], q=(0.005, 0.995), axis=0)
    ax.errorbar(lgh.wave_obs[~uplim_mask],
                delchi_mid,
                yerr=[delchi_mid-delchi_16, delchi_84-delchi_mid],
                **data_kwargs_inner)
    ax.errorbar(lgh.wave_obs[~uplim_mask],
                delchi_mid,
                yerr=[delchi_mid-delchi_05, delchi_95-delchi_mid],
                **data_kwargs_outer)
    ax.errorbar(lgh.wave_obs[~uplim_mask],
                delchi_mid,
                yerr=[delchi_mid-delchi_01, delchi_99-delchi_mid],
                marker='',
                linestyle='',
                color='k',
                alpha=0.4,
                capsize=1)

    if (Nuplims > 0):
        delchi_uplim = (uplim_sigma * lgh.Lnu_unc[None,:] - lnu_att) / lnu_unc_total
        ax.scatter(lgh.wave_obs[uplim_mask],
                   np.amax(delchi_uplim, axis=0)[uplim_mask],
                   **uplim_kwargs)

    if ((lgh.xray_stellar_em is not None) or (lgh.xray_agn_em is not None)):

        # Establish the observed wavelengths
        xray_mask = np.array(['XRAY' in s for s in lgh.filter_labels])
        if (lgh.xray_stellar_em is not None):
            xray_wave_obs = lgh.xray_stellar_em.wave_obs
        else:
            xray_wave_obs = lgh.xray_agn_em.wave_obs

        if (lgh.xray_mode == 'counts'):

            counts_xray = lgh.get_xray_model_counts(samples)
            counts_unc_total = np.sqrt(lgh.xray_counts_unc[None,:]**2 + (lgh.model_unc[None,:] * counts_xray)**2)

            delchi_xray = (lgh.xray_counts[None,:] - counts_xray) / counts_unc_total

        else:
            lnu_xray,_ = lgh.get_xray_model_lnu(samples)
            lnu_xray_unc_total = np.sqrt(lgh.Lnu_unc[None,:]**2 + (lgh.model_unc[None,:] * lnu_xray)**2)

            delchi_xray = (lgh.Lnu_obs[None,:] - lnu_xray) / lnu_xray_unc_total

            if (Nuplims > 0):
                delchi_xray_uplim = (uplim_sigma * lgh.Lnu_unc[None,:] - lnu_xray) / lnu_xray_unc_total
                ax.scatter(lgh.wave_obs[uplim_mask],
                           np.amax(delchi_xray_uplim, axis=0)[uplim_mask],
                           **uplim_kwargs)

        delchi_mid = np.nanmedian(delchi_xray[:,xray_mask & ~uplim_mask], axis=0)
        delchi_16, delchi_84 = np.nanquantile(delchi_xray[:,xray_mask & ~uplim_mask], q=(0.16, 0.84), axis=0)
        delchi_05, delchi_95 = np.nanquantile(delchi_xray[:,xray_mask & ~uplim_mask], q=(0.05, 0.95), axis=0)
        delchi_01, delchi_99 = np.nanquantile(delchi_xray[:,xray_mask & ~uplim_mask], q=(0.005, 0.995), axis=0)

        ax.errorbar(xray_wave_obs[xray_mask & ~uplim_mask],
                    delchi_mid,
                    yerr=[delchi_mid-delchi_16, delchi_84-delchi_mid],
                    **data_kwargs_inner)
        ax.errorbar(xray_wave_obs[xray_mask & ~uplim_mask],
                    delchi_mid,
                    yerr=[delchi_mid-delchi_05, delchi_95-delchi_mid],
                    **data_kwargs_outer)
        ax.errorbar(xray_wave_obs[xray_mask & ~uplim_mask],
                    delchi_mid,
                    yerr=[delchi_mid-delchi_01, delchi_99-delchi_mid],
                    marker='',
                    linestyle='',
                    color='k',
                    alpha=0.4,
                    capsize=1)


    ax.set_xscale('log')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    return fig, ax
