import numpy as np
import matplotlib.pyplot as plt

def chain_plot(lgh, samples, plot_median=True, median_color='darkorange', **kwargs):
    '''Produce a chain plot of samples from an SED fit.

    The chain plots are placed in a single matplotlib figure,
    all in a single column. Not that this may be unwieldy in cases with many
    free parameters.

    Parameters
    ----------
    lgh : lightning.Lightning object
        Used to assign names (and maybe units) to the sample
        dimensions.
    samples : np.ndarray, (Nsamples, Nparam), float
        The sampled parameters. Note that this should also include
        any constant parameters.
    plot_median : bool
        If true, draw a horizontal line at the median of the chain.
    median_color : string
        The color to plot the median line with.
    **kwargs : dict
        Keyword arguments are passed on to `plt.plot` for the chain plot.

    Outputs
    -------
    matplotlib figure and axes objects containing the chain plot.

    '''

    chain_shape = samples.shape
    Nsamples = chain_shape[0]

    const_dim = np.var(samples, axis=0) == 0
    var_dim = ~const_dim
    Nvar = np.count_nonzero(var_dim)

    param_labels = []
    for mod in lgh.model_components:
        if mod is not None:
            param_labels = param_labels + mod.param_names_fncy

    param_labels = np.array(param_labels)
    param_labels_var = param_labels[var_dim]

    samples_var = samples[:,var_dim]

    # One column with, potentially 15 plots.
    # It's not pretty, but it works, I guess. And figuring
    # out how to split them into different columns
    # is a little unpleasant
    fig, axs = plt.subplots(Nvar, 1, figsize=(5, Nvar * 1))

    tt = np.arange(Nsamples) + 1

    for i in np.arange(Nvar):

        axs[i].plot(tt, samples_var[:,i].flatten(), **kwargs)
        axs[i].set_ylabel(param_labels_var[i])

        if plot_median:
            axs[i].axhline(np.median(samples_var[:,i].flatten()), color=median_color, label='Median')

        if (i < Nvar - 1):
            axs[i].set_xticklabels([])

        axs[i].set_xlim(1, tt[-1])

    axs[-1].set_xlabel('Sample Number')
    axs[0].legend(loc='upper right', frameon=True)

    return fig, axs
