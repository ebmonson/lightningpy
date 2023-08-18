import numpy as np
from pathlib import Path
from scipy.io import readsav
from scipy.interpolate import interpn

from .base import AnalyticAtten

class Doore2021Atten(AnalyticAtten):

    type = 'analytic'
    model_name = 'Doore-2021'
    Nparams = 5
    param_names = ['d21_tauB', 'd21_Fclump', 'd21_cosi', 'd21_Rold', 'd21_BD']
    param_descr = ['Face-on B-band optical depth',
                   'Birth-cloud clumpiness factor', # Alt. "Luminosity-weighted fraction of sightlines to young stars obstructed by birth clouds"
                   'Cosine of viewing/inclination angle',
                   'Fraction of intrinsic luminosity density from the old stellar component',
                   'Intrinsic bulge-to-disk ratio']
    param_names_fncy = [r'$\tau_{B,0}$', r'$F$', r'$\cos i_{D21}$', r'$R_{\rm old}$', r'$B/D$']
    param_bounds = np.array([[0, 8.0],
                             [0, 1],
                             [0, 1],
                             [0, 1],
                             [0, 1]])

    def __init__(self, wave, path_to_models=None):

        self.wave = wave
        self.Nwave = len(self.wave)

        if (path_to_models is None):
            self.path_to_models = str(Path(__file__).parent.resolve()) + '/../models/dust/att/'
        else:
            self.path_to_models = path_to_models
            if(self.path_to_models[-1] != '/'): self.path_to_models = self.path_to_models + '/'

        coeff_path = self.path_to_models + 'd21_coeff.sav'
        integral_path = self.path_to_models + 'd21_integrals.npy'
        #self.path_to_models += 'd21_coeff.sav'

        # Keith already went to the trouble of saving the coefficients
        # for the model in a binary format
        self.coeff_dict = readsav(coeff_path)
        #print(self.coeff_dict)

        # As part of the updated scheme for this model,
        # we've pre-integrated the components of
        # the extinction curve with cosi, such that we can
        # produce an 'integrated' curve that gives us L_abs.
        with open(integral_path, 'rb') as f:
            self.disk_integral = np.load(f)
            self.thin_integral = np.load(f)
            self.bulge_integral = np.load(f)

    def evaluate(self, params):

        if (len(params.shape) == 1):
            params = params.reshape(1,-1)

        params_shape = params.shape
        Nmodels = params_shape[0]
        assert (self.Nparams == params_shape[1]), "Number of parameters must match the number (%d) expected by this model (%s)" % (self.Nparams, self.model_name)

        ob = self._check_bounds(params)
        if (np.any(ob)):
            raise ValueError('Given parameters are out of bounds for this model (%s).' % (self.model_name))

        tauB = params[:,0]
        Fclumpy = params[:,1]
        cosi = params[:,2]
        Rold = params[:,3]
        BD = params[:,4]

        Rdisk = Rold / (1 + BD)
        Rbulge = Rold - Rdisk

        # tau values used in Table 3 in Popescu et al (2011)
        # plus an added value at 0.0?
        # tau = [0.1, 0.3, 0.5, 1.0, 2.0, 4.0, 8.0]
        tau = np.array([0.0, 0.1, 0.3, 0.5, 1.0, 2.0, 4.0, 8.0])
        Ntau = len(tau)

        # Include waveband at 50000 Angstroms where mlambda is 0 for interpolation smoothness
        # Wavelengths of the bands used (Angstroms)
        wavebands = np.array([912, 1350, 1500, 1650, 2000, 2200, 2500, 2800, 3650, 4430, 5640, 8090, 12590, 22000, 50000])
        wavebands = wavebands * 1e-4
        Nbands = len(wavebands)

        # Values from Table E.4 in Popescu et al (2011)
        Fcal = 0.35
        flam_fcal_column = np.array([0.427, 0.484, 0.527, 0.545, 0.628, 0.676, 0.739, 0.794, 0.892, 0.932, 0.962, 0.984, 0.991, 0.994, 0.999])

        flam = (1 - flam_fcal_column) / Fcal

        # The polynomials are actually defined in terms of 1 - cosi
        Ncosi = 70
        cosi_hr = np.linspace(0, 1, Ncosi)
        onemcosi = 1 - cosi_hr

        onemcosi_powers = np.ones((cosi_hr, 6))
        onemcosi_powers[:,1] = onemcosi
        for i in [2,3,4,5]:
            onemcosi_powers[:,i] = onemcosi**i

        # The actual shape of the source coefficient array is (6, 14, 7) = (Ncoeff, Nbands-1, Ntau-1)
        # The two points we've added (tauB = 0 and lambda = 5e4 A) both have mlambda = 0 by definition.
        mlambda_disk = np.zeros((Ncosi, Nbands, Ntau))
        mlambda_thin = np.zeros((Ncosi, Nbands, Ntau))
        mlambda_bulge = np.zeros((Ncosi, Nbands, Ntau))

        mlambda_disk[:,:-1,1:] = np.sum(self.coeff_dict['adisk'][None,:,:,:] * onemcosi_powers[:,:,None,None], axis=1)
        mlambda_thin[:,:-1,1:] = np.sum(self.coeff_dict['atdisk'][None,:,:,:] * onemcosi_powers[:,:,None,None], axis=1)
        mlambda_bulge[:,:-1,1:] = np.sum(self.coeff_dict['abulge'][None,:,:,:] * onemcosi_powers[:,:,None,None], axis=1)

        att_bulge = Rbulge[:,None,None] * 10**(-0.4 * mlambda_bulge)
        att_disk = Rdisk[:,None,None] * 10**(-0.4 * mlambda_disk)

        att_thin_1 = (1 - Rdisk[:,None,None] - Rbulge[:,None,None]) * 10**(-0.4 * mlambda_thin)
        att_thin = (1 - (Fclumpy[:,None,None] * flam[None,:,None])) * att_thin_1

        mlambda_tot = -2.5 * np.log10(att_bulge + att_disk + att_thin)

        # These might now be nonzero after
        # the computation of the thin disk component
        mlambda_tot[:,-1,:] = 0
        mlambda_tot[:,:,0] = 0

        # Now we interpolate to the input tauB and wavelength grid;
        # # it makes sense to do a two-step 1D interpolation, wavelength first and
        # # then tau.
        # finterp_wave = interp1d(mlambda_tot, wavebands, axis=1)
        # mlambda_tot_1 = finterp_wave(self.wave) # Nmodels, Nwave, Ntau
        dm = np.zeros((Nmodels, len(self.wave)))
        for i, row in enumerate(mlambda_tot):
            coords = np.stack([self.wave, np.full(len(self.wave), tauB[i])], axis=-1)
            dm[i,:] = interpn((wavebands, tau),
                              row,
                              coords,
                              method='linear',
                              bounds_error=False,
                              fill_value=0.0).flatten()
            #print(dm)

        # The transpose here produces a result with shape (Nmodels, Nwave)

        # coords = np.outer(tauB, self.wave)
        # print(coords.shape)
        # print(mlambda_tot.shape)
        # dm = interpn((tau, wavebands),
        #              mlambda_tot.T,
        #              coords,
        #              method='linear',
        #              bounds_error=False,
        #              fill_value=0.0)

        expmtau = 10**(-0.4 * dm)

        if Nmodels == 1:
            expmtau = expmtau.flatten()

        return expmtau

    def evaluate_integrated(self, params):
        '''
        This computes the "integrated" e(-tau) attenuation curve,
        that is, int[e(-tau) d cosi]. Under the assumption that our
        source spectrum does not vary with cosi, we can then integrate
        this result against the source spectrum to calculate the absorbed
        power.
        '''

        if (len(params.shape) == 1):
            params = params.reshape(1,-1)

        params_shape = params.shape
        Nmodels = params_shape[0]
        assert (self.Nparams == params_shape[1]), "Number of parameters must match the number (%d) expected by this model (%s)" % (self.Nparams, self.model_name)

        ob = self._check_bounds(params)
        if (np.any(ob)):
            raise ValueError('Given parameters are out of bounds for this model (%s).' % (self.model_name))

        tauB = params[:,0]
        Fclumpy = params[:,1]
        cosi = params[:,2]
        Rold = params[:,3]
        BD = params[:,4]

        Rdisk = Rold / (1 + BD)
        Rbulge = Rold - Rdisk

        # tau values used in Table 3 in Popescu et al (2011)
        # plus an added value at 0.0?
        # tau = [0.1, 0.3, 0.5, 1.0, 2.0, 4.0, 8.0]
        tau = np.array([0.0, 0.1, 0.3, 0.5, 1.0, 2.0, 4.0, 8.0])
        Ntau = len(tau)

        # Include waveband at 50000 Angstroms where mlambda is 0 for interpolation smoothness
        # Wavelengths of the bands used (Angstroms)
        wavebands = np.array([912, 1350, 1500, 1650, 2000, 2200, 2500, 2800, 3650, 4430, 5640, 8090, 12590, 22000, 50000])
        Nbands = len(Nbands)

        # Values from Table E.4 in Popescu et al (2011)
        Fcal = 0.35
        flam_fcal_column = np.array([0.427, 0.484, 0.527, 0.545, 0.628, 0.676, 0.739, 0.794, 0.892, 0.932, 0.962, 0.984, 0.991, 0.994, 0.999])

        flam = (1 - flam_fcal_column) / Fcal

        att_bulge = Rbulge[:,None,None] * self.bulge_integral[None,:,:]
        att_disk = Rdisk[:,None,None] * self.disk_integral[None,:,:]

        att_thin_1 = (1 - Rdisk[:,None,None] - Rbulge[:,None,None]) * self.thin_integral[None,:,:]
        att_thin = (1 - (Fclumpy[:,None,None] * flam[None,:,None])) * att_thin_1

        mlambda_tot = -2.5 * np.log10(att_bulge + att_disk + att_thin)

        # These might now be nonzero after
        # the computation of the thin disk component
        mlambda_tot[:,-1,:] = 0
        mlambda_tot[:,:,0] = 0

        # Now we interpolate to the input tauB and wavelength grid;
        # The transpose here produces a result with shape (Nmodels, Nwave)
        coords = np.outer(tauB, self.wave)
        dm = interpn((tau, wavebands),
                     mlambda_tot.T,
                     coords,
                     method='linear',
                     bounds_error=False,
                     fill_value=0.0)

        expmtau = 10**(-0.4 * dm)

        if Nmodels == 1:
            expmtau = expmtau.flatten()

        return expmtau
