'''
    Base Calzetti attenuation and the modified Calzetti attenuation from
    Noll et al. (2009).

    TODO:
    - Update extrapolation of Calzetti curves to new method
'''

import numpy as np
from .base import AnalyticAtten

class CalzettiAtten(AnalyticAtten):
    '''
        Featureless Calzetti+(2000) attenuation curve.
    '''

    type = 'analytic'
    model_name = 'Calzetti'
    Nparams = 1
    param_names = ['calz_tauV_diff']
    param_descr = ['Optical depth of the diffuse ISM']
    param_bounds = np.array([0, np.inf]).reshape(1,2)

    def __init__(self, wave):

        self.wave = wave
        self.Nwave = len(self.wave)


    def evaluate(self, params):
        '''
            Model includes a featureless Calzetti law.
        '''

        klam = np.zeros_like(self.wave) # k_lambda, the opacity as a function of wavelength?
        flam1 = np.zeros_like(self.wave)
        flam2 = np.zeros_like(self.wave)

        # Different in this case since there is only one parameter
        if (len(params.shape) == 1):
            params = params.reshape(-1,1)

        params_shape = params.shape
        Nmodels = params_shape[0]
        assert (self.Nparams == params_shape[1]), "Number of parameters must match the number (%d) expected by this model (%s)" % (self.Nparams, self.model_name)

        ob = self.check_bounds(params)
        if (np.any(ob)):
            raise ValueError('Given parameters are out of bounds for this model (%s).' % (self.model_name))

        tauV_diff = params[:,0]

        RV = 4.05

        # Calzetti curve extrapolated from 0.01 to 5.0 microns
        # originally derived from 0.0912 to 2.2 microns
        w1 = (self.wave >= 0.6300) & (self.wave <= 2.2000)
        w2 = (self.wave >= 0.0912) & (self.wave <= 0.6300)
        w3 = (self.wave < 0.0912) & (self.wave > 2.2000)

        kk = 1./self.wave

        klam[w1] = 2.659 * (-1.857 + 1.040 * kk[w1]) + RV
        klam[w2] = 2.659 * np.polynomial.Polynomial([-2.156, 1.509, -0.198, 0.011])(kk[w2]) + RV
        klam[w3] = 0.0

        flam1 = klam[None,:] / 4.05

        expminustau = np.exp(-1 * tauV_diff[:,None] * flam1)

        if (Nmodels == 1):
            expminustau = expminustau.flatten()


        return expminustau


class ModifiedCalzettiAtten(AnalyticAtten):
    '''
        The Noll+(2009) modification of the Calzetti+(2000) attenuation curve,
        including a Drude-profile bump at 2175 Å and a variable UV slope.
    '''

    type = 'analytic'
    model_name = 'Modified-Calzetti'
    Nparams = 3
    param_names = ['mcalz_tauV_diff', 'mcalz_delta', 'mcalz_tauV_BC']
    param_descr = ['Optical depth of the diffuse ISM',
                   'Deviation from the Calzetti+2000 UV power law slope',
                   'Optical depth of the birth cloud in star forming regions']
    param_bounds = np.array([[0, np.inf],
                             [-np.inf, np.inf],
                             [0, np.inf]])

    def __init__(self, wave):

        self.wave = wave
        self.Nwave = len(self.wave)


    def evaluate(self, params):
        '''
            Model includes a featureless Calzetti law, with the
            addition of a UV bump at 2175 A and optionally extra birth cloud
            extinction. The same attenuation model used in most cases by
            Lightning; I ported it from the Lightning IDL source.

            As of right now the birth cloud component should be ignored, it isn't really
            implemented properly at the moment -- it'll be applied to all ages if you set
            tauV_BC > 0.

            If I were willing to be a little more clever, I would define this more obviously
            as an extension of the CalzettiAtten class.
        '''

        klam = np.zeros_like(self.wave)
        flam1 = np.zeros_like(self.wave)
        flam2 = np.zeros_like(self.wave)

        if (len(params.shape) == 1):
            params = params.reshape(1,-1)

        params_shape = params.shape
        Nmodels = params_shape[0]
        assert (self.Nparams == params_shape[1]), "Number of parameters must match the number (%d) expected by this model (%s)" % (self.Nparams, self.model_name)

        ob = self.check_bounds(params)
        if (np.any(ob)):
            raise ValueError('Given parameters are out of bounds for this model (%s).' % (self.model_name))

        tauV_diff = params[:,0]
        delta = params[:,1]
        tauV_BC = params[:,2]

        RV = 4.05
        FWHM_bump = 0.0350 # 350 A
        lam0_bump = 0.2175 # 2175 A
        # Modifications as in Noll et al. 2009
        Dlam = (0.85 - 1.9 * delta[:,None]) / (((self.wave[None,:]**2 - lam0_bump**2) / (self.wave[None,:] * FWHM_bump))**2 + 1.)

        # Calzetti curve extrapolated from 0.01 to 5.0 microns
        # originally derived from 0.0912 to 2.2 microns
        w1 = (self.wave >= 0.6300) & (self.wave <= 2.2000)
        w2 = (self.wave >= 0.0912) & (self.wave <= 0.6300)
        w3 = (self.wave < 0.0912) & (self.wave > 2.2000)

        kk = 1./self.wave

        klam[w1] = 2.659 * (-1.857 + 1.040 * kk[w1]) + RV
        klam[w2] = 2.659 * np.polynomial.Polynomial([-2.156, 1.509, -0.198, 0.011])(kk[w2]) + RV
        klam[w3] = 0.0

        flam1 = (klam[None,:] + Dlam) / 4.05 * (self.wave[None,:] / 0.55)**delta[:,None]

        # if (tauV_BC != 0):
        #     flam2  = 0.55 / wave # For birth cloud attenuation, somehow
        #

        flam2  = 0.55 / self.wave

        expminustau = np.exp(-1*tauV_diff[:,None] * flam1 + -1*tauV_BC[:,None] * flam2[None,:])
        if (Nmodels == 1):
            expminustau = expminustau.flatten()


        return expminustau
