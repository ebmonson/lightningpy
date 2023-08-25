import numpy as np
import h5py

import matplotlib.pyplot as plt
plt.style.use('ebm-dejavu')

from lightning import Lightning
from lightning.plots import chain_plot, corner_plot, sed_plot_bestfit, sed_plot_delchi, sfh_plot

f = h5py.File('toy_model_output.hdf5', 'r')
lgh = Lightning.from_json('toy_model_config.json')

fig, axs = chain_plot(lgh, f['mcmc/samples'])
fig.savefig('plots/chain.pdf')
plt.close(fig)

fig = corner_plot(lgh, f['mcmc/samples'], smooth=1)
fig.savefig('plots/corner.pdf')
plt.close(fig)

fig = plt.figure(figsize=(12,6))
ax1 = fig.add_axes([0.1, 0.26, 0.4, 0.64])
ax2 = fig.add_axes([0.1, 0.1, 0.4, 0.15])
ax3 = fig.add_axes([0.56, 0.1, 0.34, 0.8])

fig, ax1 = sed_plot_bestfit(lgh,
                            f['mcmc/samples'],
                            f['mcmc/logprob_samples'],
                            plot_components=True,
                            ax=ax1,
                            legend_kwargs={'loc': 'lower left', 'frameon': False})
ax1.set_xticklabels([])
fig, ax2 = sed_plot_delchi(lgh,
                           f['mcmc/samples'],
                           f['mcmc/logprob_samples'],
                           ax=ax2)
fig, ax3 = sfh_plot(lgh,
                    f['mcmc/samples'],
                    ax=ax3)

fig.savefig('plots/sed_sfh.pdf')
plt.close(fig)

f.close()
