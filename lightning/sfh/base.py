import numpy as np
from scipy.integrate import trapz, cumtrapz

#################################
# Functional SFHs
#################################
class FunctionalSFH:
    '''Base class for functional (delayed exponential, double exponential, etc.) star formation histories.

    Parameters
    ----------
    age : array-like
        Grid of stellar ages on which to evaluate the SFH.

    '''

    type = 'functional'
    model_name = None
    Nparams = None
    param_names = ['None']
    param_descr = ['None']
    param_names_fncy = [r'None']
    param_bounds = np.array([None, None])


    def __init__(self, age):

        self.age = age


    def _check_bounds(self, params):
        '''
            Check that the parameters are within the ranges where the model is
            meaningful and defined. Return the indices where the model
            is out of bounds.
        '''

        if len(params.shape) == 1:
            params = params.reshape(1,-1)

        if self.Nparams is not None:
            assert (self.Nparams == params.shape[1]), "Number of parameters must match the number (%d) expected by this model (%s)" % (self.Nparams, self.model_name)
        Nmodels = params.shape[0]

        ob_idcs = (params < self.param_bounds[:,0][None,:]) | (params > self.param_bounds[:,1][None,:])

        return ob_idcs

    def evaluate(self, params):
        '''Return the SFR as a function of time.

        Parameters
        ----------
        params : array-like (Nmodels, Nparams)
            Model parameters.

        Returns
        -------
        sfrt : array-like (Nmodels, Nages)

        '''

        if len(params.shape) == 1:
            params = params.reshape(1,-1)

        if self.Nparams is not None:
            assert (self.Nparams == params.shape[1]), "Number of parameters must match the number (%d) expected by this model (%s)" % (self.Nparams, self.model_name)
        Nmodels = params.shape[0]

        ob = self._check_bounds(params)
        if (np.any(ob)):
            raise ValueError('Given parameters are out of bounds for this model (%s).' % (self.model_name))

        sfrt = np.zeros((Nmodels, len(self.age)))

        if Nmodels == 1:
            sfrt = sfrt.flatten()

        return sfrt


    def multiply(self, params, arr):
        '''Multiply the SFR in each bin by a supplied array (to determine, e.g., the stellar mass in each bin).

        For normal internal use this function determines the appropriate broadcast shape
        of the output based on the shape of ``arr`` (which could be, e.g. stellar mass as a function of time, or
        luminosity density as a function of time *and* wavelength). If you find yourself using this function on
        its own, check the shape of the outputs carefully.

        Parameters
        ----------
        params : array-like (Nmodels, Nparams)
            Model parameters.

        arr : array-like, (..., Nbins, ...) (up to three dimensions)
            Array to multiply by the SFH.

        Returns
        -------
        res : array-like
            Product of sfh and ``arr`` broadcast to whatever shape was deemed appropriate.
        '''

        if len(params.shape) == 1:
            params = params.reshape(1,-1)

        Nmodels = params.shape[0]
        Nages = len(self.age)

        #assert (len(self.age) == arr.shape[0]), "Number of stellar age points in SFH model and stellar model must match."
        if self.Nparams is not None:
            assert (self.Nparams == params.shape[1]), "Number of parameters must match the number (%d) expected by this model (%s)" % (self.Nparams, self.model_name)

        # This is always 2D
        sfrt = self.evaluate(params).reshape(Nmodels,-1) # (Nmodels, Nages)

        # The input array might be a spectrum, with shape (Nages, Nwave),
        # or a multidimensional model with shape (Nmodels, Nages, Nwave)
        # so we handle these different cases.
        input_dims = len(arr.shape)
        if (input_dims == 1) or (input_dims == 0):
            #output_dims = 2
            #output_shape = (Nmodels, Nages)
            assert (arr.shape[0] == Nages), "Number of stellar age points in SFH model and stellar model must match."
            res = sfrt * arr[None,:]
        elif input_dims == 2:
            #output_dims = 3
            #output_shape = (Nmodels, Nages, arr.shape[-1])
            assert (arr.shape[0] == Nages), "Number of stellar age points in SFH model and stellar model must match."
            res = sfrt[:,:,None] * arr[None,:,:]
        elif input_dims == 3:
            #output_dims = 3
            #output_shape = (Nmodels, Nages, arr.shape[-1])
            assert (arr.shape == (Nmodels, Nages, arr.shape[-1]))
            res = sfrt[:,:,None] * arr
        else:
            raise ValueError('Input `arr` has too many (>3) dimensions.')

        # if (len(arr.shape) == 2):
        #     arr2d = True
        # else:
        #     arr2d = False

        # if (arr2d):
        #     res = sfrt[:, :, None] * arr[None, :, :] # (Nmodels, Nages, Nwave)
        #     #res = sfrt[:,:, None] * np.atleast_3d(arr)
        # else:
        #     res = sfrt * arr[None, :] # (Nmodels, Nages)

        return res


    def integrate(self, params, arr, cumulative=False):
        '''Multiply the SFR in each bin by a supplied array and integrate along the age axis.

        For normal internal use this function determines the appropriate broadcast shape
        of the output based on the shape of ``arr`` (which could be, e.g. stellar mass as a function of time, or
        luminosity density as a function of time *and* wavelength). If you find yourself using this function on
        its own, check the shape of the outputs carefully.

        Parameters
        ----------
        params : array-like (Nmodels, Nbins)
            SFR in each bin.

        arr : array-like, (..., Nbins, ...) (up to three dimensions)
            Array to multiply by the SFH.

        cumulative : bool
            If ``True``, return the cumulative intgral as a function of age.

        Returns
        -------
        res : array-like, (Nmodels, ...)
            Product of sfh and ``arr``, integrated along the age axis.
        '''

        if len(params.shape) == 1:
            params = params.reshape(1,-1)

        # assert (len(self.age) == arr.shape[0]), "Number of stellar age points in SFH model and stellar model must match."
        if self.Nparams is not None:
            assert (self.Nparams == params.shape[1]), "Number of parameters must match the number (%d) expected by this model (%s)" % (self.Nparams, self.model_name)

        Nmodels = params.shape[0]

        # Swapping axes so we can integrate away the first axis.
        integrand = np.swapaxes(self.multiply(params, arr), 0, 1)

        # if min_age is None:
        #     min_age = np.amin(self.age)
        # if max_age is None:
        #     max_age = np.amax(self.age)
        #
        # mask = (self.age >= min_age) & (self.age <= max_age)
        #
        # if (cumulative):
        #     res = np.zeros((Nmodels, len(self.age)))
        #     cumulative = cumtrapz(integrand[:,mask], self.age[None, mask], axis=1, initial=0)
        #     res[:, mask] = cumulative
        #     res[:, self.age > max_age] = cumulative[:,-1]
        # else:
        #     res = trapz(integrand[:,mask], self.age[None,mask], axis=1)

        if (cumulative):
            res = cumtrapz(integrand, self.age, axis=0, initial=0).T
        else:
            res = trapz(integrand, self.age, axis=0)

        if (Nmodels == 1):
            res = res.squeeze(axis=0)

        return res

#################################
# Piecewise SFH
#################################
class PiecewiseConstSFH:
    r'''Class for piecewise-constant star formation histories.

    .. math::
        \psi(t) = \psi_i,~t_i \leq t < t_{i+1}

    Parameters
    ----------
    age : array-like, (Nbins+1)
        This array should define the edges of the stellar age bins.

    '''

    type = 'piecewise'
    model_name = 'Piecewise-Constant'


    def __init__(self, age):

        self.age = age
        self.Nparams = len(age) - 1

        self.param_names = ['psi_%d' % (i + 1) for i in np.arange(self.Nparams)]
        self.param_descr = ['SFR in stellar age bin %d' % (i + 1) for i in np.arange(self.Nparams)]
        self.param_names_fncy = [r'$\psi_%d$' % (i + 1) for i in np.arange(self.Nparams)]
        self.param_bounds = np.zeros((self.Nparams, 2))
        self.param_bounds[:,0] = 0
        self.param_bounds[:,1] = np.inf


    def _check_bounds(self, params):
        '''
            Check that the parameters are within the ranges where the model is
            meaningful and defined. Return the indices where the model
            is out of bounds.
        '''

        if len(params.shape) == 1:
            params = params.reshape(1,-1)

        if self.Nparams is not None:
            assert (self.Nparams == params.shape[1]), "Number of parameters must match the number (%d) expected by this model (%s)" % (self.Nparams, self.model_name)
        Nmodels = params.shape[0]

        ob_idcs = (params < self.param_bounds[:,0][None,:]) | (params > self.param_bounds[:,1][None,:])

        return ob_idcs


    def evaluate(self, params):
        '''Return the SFR as a function of time.

        For this piecewise constant SFH, it's just a pass-through
        for `params` after checking that it's the right shape.

        Parameters
        ----------
        params : array-like (Nmodels, Nbins)
            SFR in each bin.

        Returns
        -------
        sfrt

        '''

        # Check that the model is defined for the given parameters
        ob = self._check_bounds(params)
        #ob = (np.any(params < self.param_bounds[:,0][None,:], axis=1) | np.any(params > self.param_bounds[:,1][None,:], axis=1))
        if (np.any(ob)):
            # Failing loudly vs quietly (and returning infs or nans or something)
            # for the out-of-bounds models is TBD.
            raise ValueError('Given parameters are out of bounds for this model (%s).' % (self.model_name))

        if len(params.shape) == 1:
            params = params.reshape(1,-1)

        if self.Nparams is not None:
            assert (self.Nparams == params.shape[1]), "Number of parameters must match the number (%d) expected by this model (%s)" % (self.Nparams, self.model_name)
        Nmodels = params.shape[0]

        sfrt = params

        if Nmodels == 1:
            sfrt = sfrt.flatten()

        return sfrt


    def multiply(self, params, arr):
        '''Multiply the SFR in each bin by a supplied array (to determine, e.g., the stellar mass in each bin).

        For normal internal use this function determines the appropriate broadcast shape
        of the output based on the shape of ``arr`` (which could be, e.g. stellar mass as a function of time, or
        luminosity density as a function of time *and* wavelength). If you find yourself using this function on
        its own, check the shape of the outputs carefully.

        Parameters
        ----------
        params : array-like (Nmodels, Nbins)
            SFR in each bin.

        arr : array-like, (..., Nbins, ...) (up to three dimensions)
            Array to multiply by the SFH.

        Returns
        -------
        res : array-like
            Product of sfh and ``arr`` broadcast to whatever shape was deemed appropriate.
        '''

        if len(params.shape) == 1:
            params = params.reshape(1,-1)

        Nmodels = params.shape[0]
        Nages = len(self.age) - 1

        #assert (len(self.age) - 1 == arr.shape[0]), "Number of stellar age points in SFH model and stellar model must match."
        if self.Nparams is not None:
            assert (self.Nparams == params.shape[1]), "Number of parameters must match the number (%d) expected by this model (%s)" % (self.Nparams, self.model_name)

        # This is always 2D
        sfrt = self.evaluate(params).reshape(Nmodels,-1) # (Nmodels, Nages)

        # The input array might be a spectrum, with shape (Nages, Nwave),
        # or a multidimensional model with shape (Nmodels, Nages, Nwave)
        # so we handle these different cases.

        # What about if the input is just (Nmodels, Nages)?
        # For some reason I didn't imagine that possibility.
        # When it comes up in the X-ray model I just handle it bespoke.

        input_dims = len(arr.shape)
        if (input_dims == 1) or (input_dims == 0):
            #output_dims = 2
            #output_shape = (Nmodels, Nages)
            assert (arr.shape[0] == Nages),"Number of stellar age points in SFH model and stellar model must match."
            res = sfrt * arr[None,:]
        elif input_dims == 2:
            #output_dims = 3
            #output_shape = (Nmodels, Nages, arr.shape[-1])
            assert (arr.shape[0] == Nages),"Number of stellar age points in SFH model and stellar model must match."
            res = sfrt[:,:,None] * arr[None,:,:]
        elif input_dims == 3:
            #output_dims = 3
            #output_shape = (Nmodels, Nages, arr.shape[-1])
            assert (arr.shape == (Nmodels, Nages, arr.shape[-1]))
            res = sfrt[:,:,None] * arr
        else:
            raise ValueError('Input `arr` has too many (>3) dimensions.')

        # # The input array might be a spectrum, with shape (Nages, Nwave)
        # if (len(arr.shape) == 2):
        #     arr2d = True
        # else:
        #     arr2d = False
        #
        # Nmodels = params.shape[0]
        #
        # sfrt = self.evaluate(params).reshape(Nmodels,-1) # (Nmodels, Nages)
        #
        # if (arr2d):
        #     res = sfrt[:, :, None] * arr[None, :, :]
        # else:
        #     res = sfrt * arr[None, :]

        return res


    def sum(self, params, arr, cumulative=False):
        '''Multiply the SFR in each bin by a supplied array and sum along the age axis.

        For normal internal use this function determines the appropriate broadcast shape
        of the output based on the shape of ``arr`` (which could be, e.g. stellar mass as a function of time, or
        luminosity density as a function of time *and* wavelength). If you find yourself using this function on
        its own, check the shape of the outputs carefully.

        Parameters
        ----------
        params : array-like (Nmodels, Nbins)
            SFR in each bin.

        arr : array-like, (..., Nbins, ...) (up to three dimensions)
            Array to multiply by the SFH.

        cumulative : bool
            If ``True``, return the cumulative sum as a function of age.

        Returns
        -------
        res : array-like, (Nmodels, ...)
            Product of sfh and ``arr``, summed along the age axis.
        '''

        if len(params.shape) == 1:
            params = params.reshape(1,-1)

        #assert (len(self.age) - 1 == arr.shape[0]), "Number of stellar age bins in SFH model and stellar model must match."
        if self.Nparams is not None:
            assert (self.Nparams == params.shape[1]), "Number of parameters provided (%d) must match the number (%d) expected by this model (%s)" % (params.shape[1], self.Nparams, self.model_name)
        # # The input array might be a spectrum, with shape (Nages, Nwave)
        # if (len(arr.shape) == 2):
        #     arr2d = True
        # else:
        #     arr2d = False

        Nmodels = params.shape[0]

        summand = np.swapaxes(self.multiply(params, arr), 0, 1)

        if (cumulative):
            res = np.nancumsum(summand, axis=0).T
        else:
            res = np.nansum(summand, axis=0)

        if (Nmodels == 1):
            res = res.squeeze(axis=0)

        return res
