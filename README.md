# lightning.py

> [!Note]
>
> The only stellar population models included in this distribution are the "legacy"
> PEGASE models used in IDL Lightning. The recently developed BPASS + Cloudy
> and PEGASE + Cloudy models used in Lehmer+(2024) must be [downloaded](https://www.dropbox.com/scl/fo/is74ra0tc1t0jdo4dsntm/ADDNjrtxro2euqCWmYrCO0Y?rlkey=9v113nb8rqgl5zul6xawuwdde&st=kzgq6kxr&dl=0) before they
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
Doore+(2021) inclination-dependent attenuation model. Users interested in the properties of highly-inclined
disk galaxies are encouraged to continue to use IDL `Lightning` for their analyses, and to let the authors know
if it would be nice to access that model through `lightning.py`.

## Installation
------
Currently, `lightning.py` is not available on [PyPI](https://pypi.org/) or [conda-forge](https://conda-forge.org/).
Until it is uploaded, `lightning.py` and its dependencies can be installed by cloning the repo, creating a conda environment with the required dependencies, and then installing the package:

```sh
git clone https://github.com/ebmonson/plightning.git
cd plightning
conda env create -f environment.yml
conda activate lightning
pip install .  --no-deps
```

## Documentation
---
Online documentation for the package can be found on [its readthedocs page](https://github.com/ebmonson/plightning),
and a compiled PDF version of the documentation is available [in this repository](https://github.com/ebmonson/plightning/blob/main/docs/lightningpy.pdf).


## License
---
`lightning.py` is available under the terms of the MIT license.

## Citation
---
A paper describing `lightning.py` in application is forthcoming by Monson et al.; this page will be updated on submission. In the meantime, work using the new PEGASE+Cloudy and BPASS+Cloudy models are encouraged to also cite Lehmer et al. (2024), while work using models described in Doore et al. (2023) should also cite that paper:

```tex
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
