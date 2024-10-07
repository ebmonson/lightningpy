# Custom BPASS Cloudy Models
---

Files you should put in this folder
```
BPASS_imf_chab300_fullgrid_g.h5
BPASS_imf_chab300_fullgrid_ng.h5
```

## Cloudy Configuration

Electron density: lognH = [2,3.5]
Not a free parameter, but can be selected with the `lognH` keyword.

ionization parameters, logU = [-1.5,-2.0,-2.5,-3.0,-3.5,-4.0]

grains : on ('g') or off ('ng')

Metallicity ranging from 12 + log(O/H) = 6.5 to 8.9 with step 0.1 where solar = 8.69.
Stellar metallicity set to the same as gas metallicity, with Zsolar = 0.020.

Open geometry

---
The IMF is the default BPASS chab300 IMF, a Chabrier+(2003) IMF from
0.1-300 solar masses.
