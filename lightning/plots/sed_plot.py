import numpy as np
import matplotlib.pyplot as plt

def sed_plot_bestfit(lgh, samples, logprob_samples, plot_components=False, ax=None,
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
                     legend_kwargs={'loc':'upper right', 'frameon':False}
                     ):
    '''
    Best-fit SED plot.
    '''

    bestfit = np.argmax(logprob_samples)
    bestfit_samples = samples[bestfit,:]

    lnu_hires_att_best, lnu_hires_unatt_best = lgh.get_model_lnu_hires(bestfit_samples)
    lnu_att_best, lnu_unatt_best = lgh.get_model_lnu(bestfit_samples)

    lnu_unc_total = np.sqrt(lgh.Lnu_unc**2 + (lgh.model_unc * lnu_att_best)**2)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6,5))
    else:
        fig = ax.get_figure()

    if plot_components:
        # Oh boy that's a mouthful of a function name
        lnu_components = lgh.get_model_components_lnu_hires(bestfit_samples)

        ax.plot(lgh.wave_grid_obs, lgh.nu_grid_obs * lnu_components['stellar_unattenuated'], **stellar_unatt_kwargs)
        ax.plot(lgh.wave_grid_obs, lgh.nu_grid_obs * lnu_components['stellar_attenuated'], **stellar_att_kwargs)

        if lgh.agn is not None:
            ax.plot(lgh.wave_grid_obs, lgh.nu_grid_obs * lnu_components['agn'], **agn_kwargs)

        if lgh.dust is not None:
            ax.plot(lgh.wave_grid_obs, lgh.nu_grid_obs * lnu_components['dust'], **dust_kwargs)


    ax.plot(lgh.wave_grid_obs, lgh.nu_grid_obs * lnu_hires_att_best,
            **total_kwargs)

    ax.errorbar(lgh.wave_obs,
                lgh.nu_obs * lgh.Lnu_obs,
                yerr=lgh.nu_obs * lnu_unc_total,
                **data_kwargs)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    if show_legend:
        ax.legend(**legend_kwargs)

    return fig, ax

def sed_plot_delchi(lgh, samples, logprob_samples, ax=None,
                    xlim=(0.1, 1200), ylim=(-5.1, 5.1),
                    lines=[0,-1,1], linecolors=['slategray'], linestyles=['-','--','--'],
                    xlabel=r'Observed-Frame Wavelength $[\rm \mu m]$',
                    ylabel=r'Residual $[\sigma]$',
                    data_kwargs={'marker':'D', 'color':'k', 'markerfacecolor':'k', 'capsize':2, 'linestyle':'', 'label': 'Data'}
                    ):
    '''
    Delchi residuals for the best-fit SED.
    '''

    bestfit = np.argmax(logprob_samples)
    bestfit_samples = samples[bestfit,:]

    lnu_att_best, lnu_unatt_best = lgh.get_model_lnu(bestfit_samples)

    lnu_unc_total = np.sqrt(lgh.Lnu_unc**2 + (lgh.model_unc * lnu_att_best)**2)

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

    ax.errorbar(lgh.wave_obs,
                delchi,
                yerr=(1 + np.zeros_like(delchi)),
                **data_kwargs)

    ax.set_xscale('log')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    return fig, ax
