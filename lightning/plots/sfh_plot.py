import numpy as np
import matplotlib.pyplot as plt
from .band import ModelBand
from .step_curve import step_curve

def sfh_plot(lgh, samples,
             xlabel=r'Lookback Time $t$ [yr]',
             ylabel=r'SFR$(t)$ $[\rm M_{\odot}\ yr^{-1}]$',
             ax=None,
             shade_band=True,
             shade_q=(0.16 ,0.84),
             shade_kwargs={'color':'k', 'alpha':0.3, 'zorder':0},
             plot_line=True,
             line_q=0.5,
             line_kwargs={'color':'k', 'zorder':1},
             ylim=None
             ):
    '''A plot of the posterior SFR as a function of time.

    The default behavior is to plot the median of the posterior
    along with the shaded 16th to 84th percentiles.

    Parameters
    ----------
    lgh : lightning.Lightning object
        Used to assign names (and maybe units) to the sample
        dimensions.
    samples : np.ndarray, (Nsamples, Nparam), float
        The sampled parameters. Note that this should also include
        any constant parameters.
    xlabel : str
    ylabel : str
    ax : matplotlib.axes.Axes
        Axes to draw the plot in. (Default: None)
    shade_band : bool
        Should the uncertainty band be shaded?
    shade_q : tuple
        Quantiles to shade between. (Default: (0.16, 0.84))
    shade_kwargs : dict
    plot_line : bool
        Should a line be plotted?
    line_q :
        Quantile to plot the line at (Default: 0.5)
    line_kwargs : dict
        Each of the above ``*_kwargs`` parameters is a dict containing keyword arguments describing the color, style,
        label, etc. of the corresponding plot element, passed through to the appropriate ``matplotlib`` function.
    ylim : tuple

    Returns
    -------
    fig : Matplotlib figure containing the plot
    ax : Axes containing the plot
    
    '''

    sfh_type = lgh.sfh.type

    sfh_param = samples[:,0:lgh.sfh.Nparams]

    if sfh_type == 'piecewise':
        sfr_bins = lgh.sfh.evaluate(sfh_param)
        t, sfrt = step_curve(lgh.sfh.age, sfr_bins, anchor=False)
    else:
        t = lgh.sfh.age
        sfrt = lgh.sfh.evaluate(sfh_param)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6,5))
    else:
        fig = ax.get_figure()

    band = ModelBand(t)

    for mod in sfrt:
        band.add(mod)

    if shade_band:
        band.shade(shade_q, ax=ax, **shade_kwargs)

    if plot_line:
        band.line(line_q, ax=ax, **line_kwargs)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.set_xlim(1e6, t[-1])
    ax.set_xscale('log')
    ax.set_ylim(ylim)

    return fig, ax
