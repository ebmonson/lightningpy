.. lightning.py documentation master file, created by
   sphinx-quickstart on Thu Mar 30 15:45:30 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Lightning
=========

``lightning.py`` is the python version of the Lightning SED-fitting code.
The goal of the project is to maintain Lightning's primary design philosophy
of simplicity and physically-based models, while taking advantage of Python's
object-oriented nature and the wide array of pre-existing astronomical Python
code to improve modularity and user-friendliness.

The IDL version of ``Lightning`` is now considered the "legacy" version, and will not
see further development. Its documentation will remain available at `lightning-sed.readthedocs.io <https://lightning-sed.readthedocs.io>`_,
and the source code can still be downloaded from `github.com/rafaeleufrasio/lightning <https://www.github.com/rafaeleufrasio/lightning>`_.

This new python version contains all the features of IDL ``Lightning`` with the current notable exception of the
`Doore et al. (2021) <https://ui.adsabs.harvard.edu/abs/2021ApJ...923...26D/abstract>`_ inclination-dependent attenuation model. Users interested in the properties of highly-inclined
disk galaxies are encouraged to continue to use IDL ``Lightning`` for their analyses, and to let the authors know
if it would be nice to access that model through ``lightning.py``.

.. toctree::
   :maxdepth: 1
   :caption: Documentation

   installation
   getting_started.ipynb
   model_choice
   solver_choice
   catalog_postprocessing
   examples
   model_library
   nebular_emission_recipe
   filter_profiles
   faq
   API_reference

Attribution
===========
A paper describing ``lightning.py`` is forthcoming by Monson et al.; this page will be updated on submission. In the
meantime, work using the new PEGASE+Cloudy and BPASS+Cloudy models are encouraged to also cite `Lehmer et al. (2024, in press at ApJS) <https://ui.adsabs.harvard.edu/abs/2024arXiv241019901L/abstract>`_::

    @ARTICLE{2024arXiv241019901L,
        author = {{Lehmer}, Bret D. and {Monson}, Erik B. and {Eufrasio}, Rafael T. and {Amiri}, Amirnezam and {Doore}, Keith and {Basu-Zych}, Antara and {Garofali}, Kristen and {Oskinova}, Lidia and {Andrews}, Jeff J. and {Antoniou}, Vallia and {Geda}, Robel and {Greene}, Jenny E. and {Kovlakas}, Konstantinos and {Lazzarini}, Margaret and {Richardson}, Chris T.},
        title = "{An Empirical Framework Characterizing the Metallicity and Star-Formation History Dependence of X-ray Binary Population Formation and Emission in Galaxies}",
        journal = {arXiv e-prints},
        keywords = {Astrophysics - Astrophysics of Galaxies, Astrophysics - High Energy Astrophysical Phenomena},
        year = 2024,
        month = oct,
        eid = {arXiv:2410.19901},
        pages = {arXiv:2410.19901},
        archivePrefix = {arXiv},
        eprint = {2410.19901},
        primaryClass = {astro-ph.GA},
        adsurl = {https://ui.adsabs.harvard.edu/abs/2024arXiv241019901L},
        adsnote = {Provided by the SAO/NASA Astrophysics Data System}
    }

Work using models described in `Doore et al. (2023) <https://ui.adsabs.harvard.edu/abs/2023ApJS..266...39D/abstract>`_ should also cite that paper::

    @ARTICLE{2023ApJS..266...39D,
        author = {{Doore}, Keith and {Monson}, Erik B. and {Eufrasio}, Rafael T. and {Lehmer}, Bret D. and {Garofali}, Kristen and {Basu-Zych}, Antara},
        title = "{Lightning: An X-Ray to Submillimeter Galaxy SED-fitting Code with Physically Motivated Stellar, Dust, and AGN Models}",
        journal = {\apjs},
        keywords = {Extragalactic astronomy, Galaxy properties, Star formation, Spectral energy distribution, 506, 615, 1569, 2129, Astrophysics - Astrophysics of Galaxies},
        year = 2023,
        month = jun,
        volume = {266},
        number = {2},
        eid = {39},
        pages = {39},
        doi = {10.3847/1538-4365/accc29},
        archivePrefix = {arXiv},
        eprint = {2304.06753},
        primaryClass = {astro-ph.GA},
        adsurl = {https://ui.adsabs.harvard.edu/abs/2023ApJS..266...39D},
        adsnote = {Provided by the SAO/NASA Astrophysics Data System}
    }

License
=======
``lightning.py`` is available under the terms of the MIT license.

Acknowledgments
===============
``lightning.py`` has been developed by Erik B. Monson, Amirnezam Amiri, Keith
Doore, and Rafael Eufrasio. Much of ``lightning.py`` is ported from or based on code from
IDL ``Lightning``, which was developed by Rafael Eufrasio, Keith Doore, and Erik B. Monson.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
