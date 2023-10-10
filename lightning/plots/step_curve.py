import numpy as np

def step_curve(bin_edges, y_values, anchor=False):
    '''From a histogram (bin edges and count values) generate a step curve for
    use with matplotlib's ``plot`` function.

    This function takes an array of N + 1 bin edges and
    an array of N values and turns them into a list of 2N
    vertices so that you can make a step plot with the ``plot`` command
    rather than ``step``, ``hlines`` and ``vlines``, etc.

    Parameters
    ----------
    bin_edges : numpy array, (N+1), float
        Array defining the edges of the histogram bins
        (i.e., the locations of the steps). Assumed to
        be a monotonically increasing sequence.
    y_values : numpy array, (..., N), float
        Array containing the y values corresponding to
        bin_edges: the value of the function on
        (bin_edges[i], bin_edges[i+1]) is y_values[...,i].
        Note that this can be a 2D array, where the first axis
        cycles among different curves.
    anchor : bool (default False)
        If True, the returned curve will be anchored to
        0 on both sides.

    Outputs
    -------
    Two numpy arrays containing the x- and y-values of the
    step curve, respectively.

    Notes
    -----
    If anchor is False, each curve has 2N points, where N is the
    number of y-values.

    If anchor is True, each curve has 2N + 2 points.

    This function actually doesn't do any plotting, it just makes
    arrays to feed into ``plot``.

    '''

    bin_edges = np.array(bin_edges)
    y_values = np.array(y_values)

    if len(y_values.shape) < 2:
        y_values = y_values.reshape(1,-1)

    Ncurves = y_values.shape[0]
    Npoints = y_values.shape[1]

    out_x_values = np.zeros(2*Npoints, dtype='float')
    out_y_values = np.zeros((Ncurves, 2*Npoints), dtype='float')
    for i in np.arange(Npoints):
        out_x_values[2*i] = bin_edges[i]
        out_x_values[2*i + 1] = bin_edges[i + 1]
        out_y_values[:,2*i] = y_values[:,i]
        out_y_values[:,2*i + 1] = y_values[:,i]

    if(anchor):
        anchor_out_x_values = np.zeros(2*Npoints + 2, dtype='float')
        anchor_out_y_values = np.zeros((Ncurves, 2*Npoints + 2), dtype='float')
        anchor_out_x_values[0] = bin_edges[0]
        anchor_out_x_values[-1] = bin_edges[-1]
        anchor_out_y_values[:,0] = 0.0
        anchor_out_y_values[:,-1] = 0.0
        anchor_out_x_values[1:-1] = out_x_values
        anchor_out_y_values[:,1:-1] = out_y_values
        if Ncurves == 1: anchor_out_y_values = anchor_out_y_values.flatten()
        return anchor_out_x_values, anchor_out_y_values
    else:
        if Ncurves == 1: out_y_values = out_y_values.flatten()
        return out_x_values, out_y_values
