# lightning.py

[![Documentation Status](https://readthedocs.org/projects/lightningpy/badge/?version=latest)](https://lightningpy.readthedocs.io/en/latest/?badge=latest) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18011894.svg)](https://doi.org/10.5281/zenodo.18011894)

> [!Note]
>
> The only stellar population models included in this distribution are the "legacy"
> PEGASE models used in IDL Lightning. The BPASS + Cloudy models produced by the BPASS collaboration and the recently developed BPASS + Cloudy
> and PEGASE + Cloudy models used in Lehmer+(2024) must be [downloaded](https://www.dropbox.com/scl/fo/is74ra0tc1t0jdo4dsntm/ADDNjrtxro2euqCWmYrCO0Y?rlkey=9v113nb8rqgl5zul6xawuwdde&st=kzgq6kxr&dl=0) and placed in the appropriate model directories before they
> can be used with Lightning.

`lightning.py` is the python version of the Lightning SED-fitting code.
The goal of the project is to maintain Lightning's primary design philosophy
of simplicity and physically-based models, while taking advantage of Python's
object-oriented nature and the wide array of pre-existing astronomical Python
code to improve modularity and user-friendliness.

The IDL version of `Lightning` is now considered the "legacy" version, and will not
see further development. Its documentation will remain available at [lightning-sed.readthedocs.io](https://lightning-sed.readthedocs.io),
and the source code can still be downloaded from [github.com/rafaeleufrasio/lightning](https://www.github.com/rafaeleufrasio/lightning).

This new python version contains all the features of IDL `Lightning` with the current notable exception of the
[Doore et al. (2021)](https://ui.adsabs.harvard.edu/abs/2021ApJ...923...26D/abstract) inclination-dependent attenuation model. Users interested in the properties of highly-inclined
disk galaxies are encouraged to continue to use IDL `Lightning` for their analyses, and to let the authors know
if it would be nice to access that model through `lightning.py`.

## Installation
Currently, `lightning.py` is not available on [PyPI](https://pypi.org/) or [conda-forge](https://conda-forge.org/).
Instead, `lightning.py` and its dependencies can be installed by cloning the repo, creating a conda environment with the required dependencies, and then installing the package:

```sh
git clone https://github.com/ebmonson/lightningpy.git
cd lightningpy
conda env create -f environment.yml
conda activate lightning
pip install -e .  --no-deps
```

Model files will then need to be downloaded and dropped in the appropriate directories of the installed package. If you're following the instructions above, the root directory for the models will be

```
lightningpy/lightning/data/models/
```

Note that we recommend the "editable" pip install process as above, even if you have no intention of developing lightning yourself. This installation makes it easier to locate the models. If you install lightning as a static install in a conda environment, the root directory for the models will be 

```
path/to/conda/envs/lightning/lib/python3.XX/site-packages/lightning/data/models/
```

where `3.XX` should be replaced with the version of python used by your environment. If you have any doubt about which installation process you've used and the location of the lightning files, you can locate your lightning install by opening a python terminal and doing

```python
>>> import lightning
>>> print(lightning.__file__)
```

which will show you the root of the lightning install.

## Documentation
Online documentation for the package can be found on [its readthedocs page](https://lightningpy.readthedocs.io/en/latest/),
and a compiled PDF version of the documentation is available [in this repository](https://github.com/ebmonson/lightningpy/blob/main/docs/lightningpy.pdf).


## License
`lightning.py` is available under the terms of the MIT license.

## Citation(s)
`lightning.py` has an associated DOI via Zenodo, which can be used to cite the code directly:

```bibtex
@software{erik_monson_2025_18011894,
    author       = {Erik Monson and
                    Keith Doore and
                    Amiri, Amirnezam and
                    Eufrasio, Rafael and
                    Lehmer, Bret},
    title        = {lightning.py: Python SED Fitting Code},
    month        = dec,
    year         = 2025,
    publisher    = {Zenodo},
    version      = {v2025.1.0},
    doi          = {10.5281/zenodo.18011894},
    url          = {https://doi.org/10.5281/zenodo.18011894},
    swhid        = {swh:1:dir:bab41cceb1cf240e0de716fe3904359003e0a225
                    ;origin=https://doi.org/10.5281/zenodo.18011893;vi
                    sit=swh:1:snp:2a8878f1a0506a3477f635d746ee9a75f2a3
                    d5de;anchor=swh:1:rel:7d4578cc66e286870fce283ce1f1
                    6739e7216f7c;path=ebmonson-lightningpy-ed81d64
                    },
}
```

The first full release of `lightning.py` was attached to [Monson et al. (2026)](https://ui.adsabs.harvard.edu/abs/2026ApJ..1001...42M/abstract):

```bibtex
@ARTICLE{2026ApJ..1001...42M,
    author = {{Monson}, Erik B. and {Lehmer}, Bret D. and {Amiri}, Amirnezam and {Barboza}, Karina and {Barnes}, Ashley T. and {Basu-Zych}, Antara R. and {Dale}, Daniel A. and {Das}, Sanskriti and {Dlamini}, Simthembile and {Glover}, Simon and {Kreckel}, Kathryn and {Lopez}, Laura A. and {Lopez}, Sebastian and {Mathur}, Smita and {Pan}, Hsi-An and {Rodriguez}, Jennifer A. and {Sandstrom}, Karin and {Sarbadhicary}, Sumit K. and {Sun}, Jiayi and {Williams}, Thomas G.},
    title = "{Constraining the Subgalactic Relationship between Star Formation and the Hot Interstellar Medium in NGC 4254}",
    journal = {\apj},
    keywords = {Interstellar medium, X-ray astronomy, Disk galaxies, 847, 1810, 391, Astrophysics of Galaxies},
    year = 2026,
    month = apr,
    volume = {1001},
    number = {1},
    eid = {42},
    pages = {42},
    doi = {10.3847/1538-4357/ae4ecb},
    archivePrefix = {arXiv},
    eprint = {2602.20397},
    primaryClass = {astro-ph.GA},
    adsurl = {https://ui.adsabs.harvard.edu/abs/2026ApJ..1001...42M},
    adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

Work using the new PEGASE+Cloudy and BPASS+Cloudy models are encouraged to also cite [Lehmer et al. (2024)](https://ui.adsabs.harvard.edu/abs/2024arXiv241019901L/abstract), which describes the recipe for adding nebular emission to these models:

```bibtex
@ARTICLE{2024ApJ...977..189L,
    author = {{Lehmer}, Bret D. and {Monson}, Erik B. and {Eufrasio}, Rafael T. and {Amiri}, Amirnezam and {Doore}, Keith and {Basu-Zych}, Antara and {Garofali}, Kristen and {Oskinova}, Lidia and {Andrews}, Jeff J. and {Antoniou}, Vallia and {Geda}, Robel and {Greene}, Jenny E. and {Kovlakas}, Konstantinos and {Lazzarini}, Margaret and {Richardson}, Chris T.},
    title = "{An Empirical Framework Characterizing the Metallicity and Star-formation History Dependence of X-Ray Binary Population Formation and Emission in Galaxies}",
    journal = {\apj},
    keywords = {X-ray binary stars, Stellar evolutionary models, Galaxy evolution, Star formation, Spectral energy distribution, X-ray astronomy, 1811, 2046, 594, 1569, 2129, 1810, Astrophysics - Astrophysics of Galaxies, Astrophysics - High Energy Astrophysical Phenomena},
    year = 2024,
    month = dec,
    volume = {977},
    number = {2},
    eid = {189},
    pages = {189},
    doi = {10.3847/1538-4357/ad8de7},
    archivePrefix = {arXiv},
    eprint = {2410.19901},
    primaryClass = {astro-ph.GA},
    adsurl = {https://ui.adsabs.harvard.edu/abs/2024ApJ...977..189L},
    adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

Work using models described in the IDL Lightning implementation paper, [Doore et al. (2023)](https://ui.adsabs.harvard.edu/abs/2023ApJS..266...39D/abstract), should also cite that paper:

```bibtex
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
```
