import numpy as np
from scipy.integrate import trapz, cumtrapz

#################################
# Functional SFHs
#################################
class FunctionalSFH:
    '''
        Base class for functional (delayed exponential, double exponential, etc.)
        star formation histories.
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
        '''
            This method should return the SFR as a function of time
            evaluated at the age grid of the SFH model, given the supplied
            parameters.

            It must be overwritten by each specific SFH model, and it should return
            an (Nmodels, Nages) array.
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
        '''
            Evaluate the SFH for the given parameters and
            multiply it by the given array.
        '''

        if len(params.shape) == 1:
            params = params.reshape(1,-1)

        assert (len(self.age) == arr.shape[0]), "Number of stellar age points in SFH model and stellar model must match."
        if self.Nparams is not None:
            assert (self.Nparams == params.shape[1]), "Number of parameters must match the number (%d) expected by this model (%s)" % (self.Nparams, self.model_name)
        # The input array might be a spectrum, with shape (Nages, Nwave)
        if (len(arr.shape) == 2):
            arr2d = True
        else:
            arr2d = False

        Nmodels = params.shape[0]

        sfrt = self.evaluate(params).reshape(Nmodels,-1) # (Nmodels, Nages)

        if (arr2d):
            res = sfrt[:, :, None] * arr[None, :, :]
        else:
            res = sfrt * arr[None, :]

        return res


    def integrate(self, params, arr, min_age=None, max_age=None, cumulative=False):
        '''
            All-purpose integration function for integrating the spectrum,
            stellar mass, etc. of a stellar population model.
        '''

        if len(params.shape) == 1:
            params = params.reshape(1,-1)

        assert (len(self.age) == arr.shape[0]), "Number of stellar age points in SFH model and stellar model must match."
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
    '''
        Class for piecewise-constant star formation histories.
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
        '''
            This method should return the SFR as a function of time
            evaluated at the age grid of the SFH model, given the supplied
            parameters. For the piecewise constant SFH, it's just a pass-through
            for `params` after checking that it's the right shape.

            It will be overwritten by each specific SFH model, but it should return
            an (Nmodels, Nages) array.
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
        '''
            Evaluate the SFH for the given parameters and
            multiply it by the given array.
        '''

        if len(params.shape) == 1:
            params = params.reshape(1,-1)

        assert (len(self.age) - 1 == arr.shape[0]), "Number of stellar age points in SFH model and stellar model must match."
        if self.Nparams is not None:
            assert (self.Nparams == params.shape[1]), "Number of parameters must match the number (%d) expected by this model (%s)" % (self.Nparams, self.model_name)
        # The input array might be a spectrum, with shape (Nages, Nwave)
        if (len(arr.shape) == 2):
            arr2d = True
        else:
            arr2d = False

        Nmodels = params.shape[0]

        sfrt = self.evaluate(params).reshape(Nmodels,-1) # (Nmodels, Nages)

        if (arr2d):
            res = sfrt[:, :, None] * arr[None, :, :]
        else:
            res = sfrt * arr[None, :]

        return res


    def sum(self, params, arr, min_age=None, max_age=None, cumulative=False):
        '''
            All-purpose function for weighting and summing the spectrum,
            stellar mass, etc. of a stellar population model.
        '''

        if len(params.shape) == 1:
            params = params.reshape(1,-1)

        assert (len(self.age) - 1 == arr.shape[0]), "Number of stellar age bins in SFH model and stellar model must match."
        if self.Nparams is not None:
            assert (self.Nparams == params.shape[1]), "Number of parameters provided (%d) must match the number (%d) expected by this model (%s)" % (params.shape[1], self.Nparams, self.model_name)
        # The input array might be a spectrum, with shape (Nages, Nwave)
        if (len(arr.shape) == 2):
            arr2d = True
        else:
            arr2d = False

        Nmodels = params.shape[0]

        summand = np.swapaxes(self.multiply(params, arr), 0, 1)

        if (cumulative):
            res = np.cumsum(summand, axis=0).T
        else:
            res = np.sum(summand, axis=0)

        if (Nmodels == 1):
            res = res.squeeze(axis=0)

        return res
