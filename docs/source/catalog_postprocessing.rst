Catalog Postprocessing
======================

For samples of galaxies or applications like galaxy mapping, it's desirable to combine the results of all the completed
SED fits into a single file. We provide a function for this: ``lightning.postprocessing.postprocess_catalog``. The
function takes lists of "result" and "model" files and collates the results into a single HDF5 catalog. The result files
are expected to be HDF5 files with a specific format.

For ``solver_mode='mcmc'``, the chains are also expected to be in HDF5 format, with the following structure::

    mcmc
    ├── logprob_samples (Nsamples)
    ├── samples (Nsamples, Nparams)
    └── autocorr (Nparams)

Whereas for ``solver_mode='mle'``, the results are expected to be formated as::

    res
    ├── bestfit (Nparams)
    └── chi2_best ()

The model files can be either JSON or pickle files, selected by the ``model_mode`` keyword. The final HDF5 catalog file
is generally structured as follows::

    └──sourcename
        ├── mcmc
        │   ├── logprob_samples (Nsamples)
        │   └── samples (Nsamples, Nparams)
        ├── parameters
        │   ├── modelname
        │   │   └── parametername
        │   │       ├── best ()
        │   │       ├── hi ()
        │   │       ├── lo ()
        │   │       └── med ()
        └── properties
            ├── filter_labels (Nfilters)
            ├── lnu (Nfilters)
            ├── lnu_unc (Nfilters)
            ├── lumdist ()
            ├── mstar
            │   ├── best ()
            │   ├── hi ()
            │   ├── lo ()
            │   └── med ()
            ├── redshift ()
            └── pvalue ()

where e.g., MCMC keys and parameter quantiles are omitted when dealing with MLE results. See :ref:`postproc-link` in
the detailed API reference for further information.
