import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

def lineflux_plot(lgh,samples,
                  plot_unatt=False,
                  unatt_kwargs={'facecolor':'dodgerblue', 'alpha':0.5, 'label':'Unattenuated lines'},
                  att_kwargs={'facecolor':'red', 'alpha':0.5, 'label':'Attenuated lines'},
                  data_kwargs={'marker':'D', 'color':'k', 'markerfacecolor':'k', 'capsize':2, 'linestyle':'', 'label': 'Data'},
                  show_legend=True,
                  legend_kwargs={'loc':'upper right', 'frameon':False},
                  ax=None):
    '''
    Plot the posteriors on the line luminosities as violins, with the data overplotted.
    As you can imagine, this is only really legible for something in the neighborhood of 10
    lines. But if you tried to fit more lines than that, you probably have bigger problems.
    As an option, you can plot the unattenuated, intrinsic line luminosities alongside the attenuated
    luminosities (which are the ones we compare to the data).
    '''

    linelum, linelum_intr = lgh.get_model_lines(samples)
    # print(np.median(linelum, axis=0))

    if ax is None:
        fig, ax = plt.subplots(figsize=(8,8))
        # Make space for the long line labels
        plt.subplots_adjust(bottom=0.15)
    else:
        fig = ax.get_figure()


    parts1 = ax.violinplot(linelum)
    for pc in parts1['bodies']:
        pc.set_facecolor(att_kwargs['facecolor'])
        # pc.set_edgecolor(att_kwargs['color'])
        pc.set_alpha(att_kwargs['alpha'])
    parts1['cbars'].set_color(att_kwargs['facecolor'])
    parts1['cmins'].set_color(att_kwargs['facecolor'])
    parts1['cmaxes'].set_color(att_kwargs['facecolor'])
    legend_elements = [Patch(**att_kwargs)]



    if plot_unatt:
        parts2 = ax.violinplot(linelum_intr)
        for pc in parts2['bodies']:
            pc.set_facecolor(unatt_kwargs['facecolor'])
            # pc.set_edgecolor('dodgerblue')
            pc.set_alpha(unatt_kwargs['alpha'])
        parts2['cbars'].set_color(unatt_kwargs['facecolor'])
        parts2['cmins'].set_color(unatt_kwargs['facecolor'])
        parts2['cmaxes'].set_color(unatt_kwargs['facecolor'])
        legend_elements += [Patch(**unatt_kwargs)]

    #lnu_unc_total = np.median(np.sqrt(lgh.Lnu_unc[None,:]**2 + (lgh.model_unc[None,:] * lnu_att)**2), axis=0)
    line_unc_total = np.median(np.sqrt(lgh.L_lines_unc[None,:]**2 + (lgh.model_unc_lines[None,:] * linelum)**2), axis=0)

    ax.errorbar(range(1, len(lgh.line_labels) + 1), lgh.L_lines, yerr=line_unc_total,
                **data_kwargs)
    legend_elements += [Line2D([],[],
                        marker=data_kwargs['marker'],
                        color=data_kwargs['color'],
                        markerfacecolor=data_kwargs['markerfacecolor'],
                        linestyle=data_kwargs['linestyle'],
                        label=data_kwargs['label'])]

    ax.set_xticks(range(1, len(lgh.line_labels) + 1), lgh.line_labels, rotation=90.0)
    ax.set_yscale('log')
    ax.set_ylabel(r'Line Luminosity [$\rm L_{\odot}$]')

    if show_legend:
        ax.legend(handles=legend_elements, loc='best')

    return fig, ax
