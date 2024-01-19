# BPASS Cloudy Models
---

BPASS models after reprocessing by CLOUDY with a ~reasonable
set of default parameters (from https://bpass.auckland.ac.nz/4.html):

## Cloudy Configuration

Electron density: 200 cm-3

ionization parameters, logU=[-1.0,-1.5,-2.0,-2.5,-3.0,-3.5,-4.0]

grains : on or off (here that's `_gr` or `_ng`)

Standard BPASS metallicity compositions (all abundances scale relative to Solar)

sphere (spherical geometry)

covering factor 1.0 linear

iterate to convergence

stop temperature 100K (stopping conditions)

stop efrac -2

All other parameters set to default.

---
The IMF is the default BPASS IMF, a two-slope Kroupa IMF from
0.1-300 solar masses.

The spectra have been re-binned to 5 Å from the default 1 Å bins. Age bins are given by their centers, except for the first bin, which covers t=0 to 10^6.05 years.

The Cloudy-processed spectra by default only cover 1-30000 Å and t=0-10^7.5 years. I've stitched them together with the corresponding BPASS source models to cover the full 1-100000 Å and t=0-10^11 years.
