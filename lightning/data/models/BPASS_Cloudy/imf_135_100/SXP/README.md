# Custom BPASS Cloudy Models Based on Garofali+(2024)
---

Files you should put in this folder
```
BPASS_G24_imf135_100_fullgrid_g.h5
BPASS_G24_imf135_100_fullgrid_ng.h5
```

## Cloudy Configuration

Electron density: lognH = [2,3.5]
Not a free parameter, but can be selected with the `lognH` keyword.

ionization parameters, logU = [-1.5,-2.0,-2.5,-3.0,-3.5,-4.0]

grains : on ('g') or off ('ng')

Metallicity ranging from 0.001 to 0.020.
Gas metallicity set to the same as stellar metallicity, where we have assumed
Zsolar = 0.020 <-> 12 + log(O/H) = 8.69

Open geometry

---
The IMF is the BPASS 135-100 IMF, from 0.1-100 solar masses.
