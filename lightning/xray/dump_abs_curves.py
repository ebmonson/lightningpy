'''
    dump_abs_curves.py

    A script to produce tabulated versions
    of common XSPEC absorption models
    (tbabs, phabs, anything else that may be added later)
    normalized to a common column density.

    Running this script requires CIAO and Sherpa.
    Note that running it is not required to use Lightning,
    and is mainly provided for reproducibility.
'''

import astropy.constants as const
import astropy.units as u
import numpy as np
from sherpa.astro.ui import *
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['xtick.direction'] = 'in'
rcParams['ytick.direction'] = 'in'
rcParams['mathtext.fontset'] = 'dejavuserif'

def main():

    hc_um = (const.h * const.c).to(u.keV * u.micron).value

    E = np.logspace(np.log10(0.01), np.log10(20), 200)
    wave = hc_um / E
    # E = np.linspace(0.1, 20, 200)
    fig, ax = plt.subplots()

    for model in ['tbabs', 'phabs']:

        # The source model is just unity times the absorption
        # model. We normalize it to NH = 1e20 cm-2,
        # evaluate it on some grid of energies, and then
        # output it as a text file. A plot is also generated
        # for an easy reference to differences.
        # Note that there is no practical difference at X-ray
        # energies, but that tbabs models the absorption of the FUV
        # by ISM dust grains (which may be more useful to future Lightning models).
        set_source('xs%s.att * const1d.const' % model)

        mdl = get_source()
        att.nH = 0.01

        expminustau = mdl(E)

        expminustau[E > 12] = 1

        ax.plot(E, E**3 * -1 * np.log(expminustau), label=model)
        #ax.plot(E, mdl(E), label=model)

        header = '%s' % model + ' - normalized to nH = 1e20 cm-2' + '\n'
        header += '1: Rest-frame wavelength [um]' + '\n'
        header += '2: Rest-frame energy [keV]' + '\n'
        header += '3: exp(-tau)'

        out_arr = np.stack([wave[::-1], E[::-1], expminustau[::-1]], axis=-1)

        np.savetxt('../models/xray/abs/%s.txt' % (model), out_arr, header=header)


    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(0.01,10)
    #ax.set_ylim(1e-4,)
    ax.set_xlabel(r'Rest-Frame Energy [keV]')
    ax.set_ylabel(r'$E^3~\tau$')
    ax.legend(loc='lower right')

    fig.savefig('../models/xray/abs/abs_models.png', dpi=300)

if __name__ == '__main__':

    main()
