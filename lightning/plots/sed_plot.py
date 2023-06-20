import numpy as np
import matplotlib.pyplot as plt

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
    '''
    Best-fit SED plot. More ~Bayesian~ SED plots TBD. Will probably
    add a similar function for producing SED plots with uncertainty
    bands. See also the ppc_sed function for producing more diagnostic
    SED plots based on posterior predictive checks.
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

            ax.errorbar(xray_wave_obs[~uplim_mask][xray_mask],
                        xray_nu_obs[~uplim_mask][xray_mask] * lgh.Lnu_obs[~uplim_mask][xray_mask],
                        yerr=xray_nu_obs[~uplim_mask][xray_mask] * lgh.Lnu_unc[~uplim_mask][xray_mask],
                        **data_kwargs)

            ax.scatter(xray_wave_obs[uplim_mask][xray_mask],
                       uplim_sigma * xray_nu_obs[uplim_mask][xray_mask] * lgh.Lnu_unc[uplim_mask][xray_mask],
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
    '''
    Delchi residuals for the best-fit SED.
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
