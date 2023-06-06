import numpy as np
import corner

def corner_plot(lgh, samples, **kwargs):
    '''Produce a corner plot of samples from an SED fit.

    This is just a thin wrapper around corner that
    just figures out which dimensions are constant
    and assigns parameter labels to the chains.

    Parameters
    ----------
    lgh : lightning.Lightning object
        Used to assign names (and maybe units) to the sample
        dimensions.
    samples : np.ndarray, (Nsamples, Nparam), float
        The sampled parameters. Note that this should also include
        any constant parameters.
    **kwargs : dict
        Keyword arguments are passed on to `corner.corner`.

    Outputs
    -------
    matplotlib figure object containing the corner plot.

    '''

    const_dim = np.var(samples, axis=0) < 1e-10
    var_dim = ~const_dim
    Nvar = np.count_nonzero(var_dim)

    param_labels = []
    for mod in lgh.model_components:
        if mod is not None:
            param_labels = param_labels + mod.param_names_fncy

    param_labels = np.array(param_labels)
    param_labels_var = param_labels[var_dim]

    samples_var = samples[:,var_dim]

    fig = corner.corner(samples_var, labels=param_labels_var, **kwargs)

    return fig
