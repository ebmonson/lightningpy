'''
kewley_curves.py

BPT diagrams with 4 line ratios as given in
Kewley+(2006).
'''

import numpy as np
import matplotlib.pyplot as plt

def k06_NIIplot(NIIHalpha=None, ax=None,
                seyfertcolor='firebrick', seyfertstyle='-',
                compcolor='black', compstyle='--'):
    '''[OIII]/Hbeta vs. [NII]/Halpha

    Draws into a new axis by default.

    Parameters
    ----------
    NIIHalpha : np.ndarray
        Values for [NII]/Halpha ratio (Default: ``np.linspace(-2, 1, 25)``)
    ax : matplotlib.axes.Axes
        Axes for the plot (Default: None)
    seyfertcolor : str
        Color for the Seyfert line (Default: 'firebrick')
    seyfertstyle : str
        Style for the Seyfert line (Default: '-')
    compcolor : str
        Color for the composite line (Default: 'black')
    compstyle : str
        Style for the composite line (Default: '--')

    Returns
    -------
    ax

    '''

    if ax is None:
        fig, ax = plt.subplots()

    if NIIHalpha is None:
        NIIHalpha = np.linspace(-2, 1, 25)

    compline = 0.61 / (NIIHalpha - 0.05) + 1.3
    seyfertline = 0.61 / (NIIHalpha - 0.47) + 1.19

    compdef = NIIHalpha < 0.05
    seyfertdef = NIIHalpha < 0.47

    ax.plot(NIIHalpha[compdef], compline[compdef], color=compcolor, linestyle=compstyle)
    ax.plot(NIIHalpha[seyfertdef], seyfertline[seyfertdef], color=seyfertcolor, linestyle=seyfertstyle)

    return ax

def k06_SIIplot(SIIHalpha=None, ax=None,
                seyfertcolor='firebrick', seyfertstyle='-'):
    '''[OIII]/Hbeta vs. [SII]/Halpha

    Draws into a new axis by default.

    Parameters
    ----------
    SIIHalpha : np.ndarray
        Values for [NII]/Halpha ratio (Default: ``np.linspace(-1.2, 0.8, 25)``)
    ax : matplotlib.axes.Axes
        Axes for the plot (Default: None)
    seyfertcolor : str
        Color for the Seyfert line (Default: 'firebrick')
    seyfertstyle : str
        Style for the Seyfert line (Default: '-')

    Returns
    -------
    ax

    '''

    if ax is None:
        fig, ax = plt.subplots()

    if SIIHalpha is None:
        SIIHalpha = np.linspace(-1.2, 0.8, 25)

    seyfertline = 0.72 / (SIIHalpha - 0.32) + 1.3
    seyfertdef = SIIHalpha < 0.32

    ax.plot(SIIHalpha[seyfertdef], seyfertline[seyfertdef], color=seyfertcolor, linestyle=seyfertstyle)

    return ax

def k06_OIplot(OIHalpha=None, ax=None,
               seyfertcolor='firebrick', seyfertstyle='-'):
    '''[OIII]/Hbeta vs. [OI]/Halpha

    Draws into a new axis by default.

    Parameters
    ----------
    OIHalpha : np.ndarray
        Values for [NII]/Halpha ratio (Default: ``np.linspace(-2.2, 0.0, 25)``)
    ax : matplotlib.axes.Axes
        Axes for the plot (Default: None)
    seyfertcolor : str
        Color for the Seyfert line (Default: 'firebrick')
    seyfertstyle : str
        Style for the Seyfert line (Default: '-')

    Returns
    -------
    ax

    '''

    if ax is None:
        fig, ax = plt.subplots()

    if OIHalpha is None:
        OIHalpha = np.linspace(-2.2, 0, 25)

    seyfertline = 0.73 / (OIHalpha + 0.59) + 1.33
    seyfertdef = OIHalpha < -0.59

    ax.plot(OIHalpha[seyfertdef], seyfertline[seyfertdef], color=seyfertcolor, linestyle=seyfertstyle)

    return ax
