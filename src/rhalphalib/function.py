from typing import Callable, List, Optional, Tuple
from collections import OrderedDict
import numpy as np
from scipy.special import binom
import numbers
import warnings
from .parameter import IndependentParameter, NuisanceParameter, DependentParameter
from .util import install_roofit_helpers


def matrix_bernstein(n: int):
    """
    Construct the Bernstein basis matrix for a given order n
    """
    v = np.arange(n + 1)
    bmat = np.einsum("l,lv,lv->vl", binom.outer(n, v), binom.outer(v, v), np.power(-1.0, np.subtract.outer(v, v)))
    bmat[np.greater.outer(v, v)] = 0  # v > l
    return bmat


def matrix_chebyshev(n: int):
    """
    Construct the Chebyshev basis matrix for a given order n
    """
    M = np.zeros((n + 1, n + 1))
    for nth in range(1, n + 2):
        _coef_basis = np.zeros(nth)
        _coef_basis[-1] = 1
        c = np.polynomial.Chebyshev(_coef_basis, domain=[0, 1])
        p = c.convert(kind=np.polynomial.Polynomial)
        M[nth - 1, :nth] = p.coef
    return M


def matrix_poly(n: int):
    """
    Construct the polynomial basis matrix for a given order n
    """
    return np.identity(n + 1)


def svd_transform(cov):
    _, s, v = np.linalg.svd(cov)
    _transform = np.sqrt(s)[:, None] * v
    return transform


def params_from_roofit(fitresult, param_names=None):
    install_roofit_helpers()
    names = [p.GetName() for p in fitresult.floatParsFinal()]
    means = fitresult.valueArray()
    cov = fitresult.covarianceArray()
    if param_names is not None:
        pidx = np.array([names.index(pname) for pname in param_names])
        means = means[pidx]
        cov = cov[np.ix_(pidx, pidx)]
    return means, cov


def get_parvalues(params):
    return np.vectorize(lambda p: p.value)(params)


def compute_band(obj, shape, coefficients, central):
    """
    This computation follows the linearized error propagation method used in RooAbsReal::plotOnWithErrorBand() and RooCurve::calcBandInterval()
    err(x) = F(x,a) C_ab F(x,b)
    where F(x,a) = (f(x,a+da) - f(x,a-da))/2 (numerical partial derivative) and C_ab is the correlation matrix
    """
    # compute correlation matrix
    std = np.sqrt(np.diag(obj._cov))
    corr = obj._cov / std[:, None] / std[None, :]
    # compute gradients
    plus_vars = []
    minus_vars = []
    params = obj.flat_parameters
    for i, par in enumerate(params):
        val = par.value
        err = np.sqrt(obj._cov[i, i])
        # temporarily vary param
        obj.set_par_by_name(par.name, val + err)
        plus_vars.append(obj._eval(shape, coefficients))
        obj.set_par_by_name(par.name, val - err)
        minus_vars.append(obj._eval(shape, coefficients))
        # reset to central value
        obj.set_par_by_name(par.name, val)
    # flatten
    plus_vars = np.stack(plus_vars).reshape(len(params), -1)
    minus_vars = np.stack(minus_vars).reshape(len(params), -1)
    F = (plus_vars - minus_vars) / 2.0

    # final values
    def proj(F):
        return F.T @ corr @ F

    err2 = np.apply_along_axis(proj, 1, F.T).reshape(central.shape)
    lo = central + np.sqrt(err2)
    hi = central - np.sqrt(err2)
    return lo, hi


class BasisPoly:
    """
    Construct a multidimensional polynomial

    Parameters:
        name: will be used to prefix any RooFit object names
        order: tuple of order in each dimension
        dim_names: optional, names of each dimension
        init_params: ndarray of initial params
        limits: tuple of independent parameter limits, default: (0, 10)
        coefficient_transform: callable to transform coefficients before multiplying by parameters
        square_params: if True, square the parameters before multiplying by coefficient;
            this is a way to ensure a positive definite polynomial in the Bernstein basis.
    """

    def __init__(
        self,
        name: str,
        order: Tuple[int, ...],
        dim_names: Optional[Tuple[str, ...]] = None,
        basis: str = "Bernstein",
        init_params: Optional[np.ndarray] = None,
        limits: Optional[Tuple[float, float]] = None,
        coefficient_transform: Optional[Callable] = None,
        square_params: bool = False,
    ):
        self._name = name
        if not isinstance(order, tuple):
            raise ValueError
        self._order = order
        self._basis = basis
        self._shape = tuple(n + 1 for n in order)
        if dim_names:
            if len(order) != len(dim_names):
                raise ValueError
            self._dim_names = dim_names
        else:
            self._dim_names = ["dim%d" % i for i in range(len(self._order))]
        if isinstance(init_params, np.ndarray):
            if init_params.shape != self._shape:
                raise ValueError
            self._init_params = init_params
        elif init_params is None:
            self._init_params = np.ones(shape=self._shape)
        else:
            raise ValueError
        if limits is None:
            limits = (0.0, 10.0)
        self._transform = coefficient_transform
        if square_params and self._basis != "Bernstein":
            raise ValueError("square_params only applicable for Bernstein basis")

        # Construct companion matrix for each dimension
        self._bmatrices = []
        for idim, n in enumerate(self._order):
            if self._basis == "Bernstein":
                bmat = matrix_bernstein(n)
            elif self._basis == "Chebyshev":
                bmat = matrix_chebyshev(n)
            elif self._basis == "Polynomial":
                bmat = matrix_poly(n)
            else:
                raise NotImplementedError("Basis='{}' not implemented".format(basis))
            self._bmatrices.append(bmat)

        # Construct parameter tensor
        self._params = np.full(self._shape, None)
        for ipar, initial in np.ndenumerate(self._init_params):
            param = IndependentParameter("_".join([self.name] + ["%s_par%d" % (d, i) for d, i in zip(self._dim_names, ipar)]), initial, lo=limits[0], hi=limits[1])
            if not square_params:
                self._params[ipar] = param
            else:
                paramsq = DependentParameter("_".join([self.name] + ["%s_parsq%d" % (d, i) for d, i in zip(self._dim_names, ipar)]), "{0}*{0}", param)
                self._params[ipar] = paramsq
        # covariance is not populated until after fit is performed
        self._cov = None

    @property
    def name(self):
        return self._name

    @property
    def order(self):
        return self._order

    @property
    def dim_names(self):
        return self._dim_names

    @property
    def basis(self):
        return self._basis

    @property
    def parameters(self):
        return self._params

    @property
    def flat_parameters(self):
        return self._params.reshape(-1)

    @parameters.setter
    def parameters(self, newparams):
        if not isinstance(newparams, np.ndarray):
            raise ValueError("newparams should be numpy array")
        elif newparams.shape != self._params.shape:
            raise ValueError("newparams shape does not match")
        for pnew, pold in zip(newparams.reshape(-1), self.flat_parameters):
            pnew.name = pold.name
            # probably worth caching
            if pnew.intermediate:
                pnew.intermediate = False
        self._params = newparams
        # covariance is invalidated whenever parameters are changed
        self._cov = None

    def update_from_roofit(self, fit_result):
        par_names = sorted([p for p in fit_result.floatParsFinal().contentsString().split(",") if self.name in p])
        means, cov = params_from_roofit(fit_result, par_names)
        par_results = {p: round(means[i], 3) for i, p in enumerate(par_names)}
        for par in self.flat_parameters:
            par.value = par_results[par.name]
        self._cov = cov

    def set_parvalues(self, parvalues):
        for par, new_val in zip(self.flat_parameters, parvalues):
            par.value = new_val

    def set_par_by_name(self, parname, parvalue):
        for par in self.flat_parameters:
            if par.name==parname:
                par.value = parvalue

    def coefficients(self, *xvals):
        # evaluate polynomial product tensor
        bpolyval = np.ones_like(xvals[0])
        for x, n, B in zip(xvals, self._order, self._bmatrices):
            xpow = np.power.outer(x, np.arange(n + 1))
            bpolyval = np.einsum("vl,xl,x...->x...v", B, xpow, bpolyval)

        if self._transform is not None:
            bpolyval = self._transform(bpolyval)
        return bpolyval

    def _prepare(self, *vals):
        # prepare for __call__ below
        if len(vals) != len(self._order):
            raise ValueError("Not all dimension values specified")
        xvals = []
        shape = None
        for x in vals:
            if isinstance(x, numbers.Number):
                x = np.array(x)
            if not np.all((x >= 0) & (x <= 1)):
                raise ValueError("Basis polynomials are only defined on the interval [0, 1]")
            if shape is None:
                shape = x.shape
            elif shape != x.shape:
                raise ValueError("BasisPoly: all variables must have same shape")
            xvals.append(x.flatten())

        coefficients = self.coefficients(*xvals).reshape(-1, self.flat_parameters.size)

        return xvals, shape, coefficients

    def _eval(self, shape, coefficients):
        parameters = get_parvalues(self.flat_parameters)
        nominal_vals = (parameters * coefficients).sum(axis=1).reshape(shape)
        return nominal_vals

    def __call__(self, *vals, nominal: bool = False, errorband: bool = False):
        """Evaluate the polynomial at the given values

        Parameters:
            vals: a ndarray for each dimension's values to evaluate the polynomial at
            nominal: set true to evaluate nominal polynomial (rather than create DependentParameter objects)
            errorband: set true to output error band along with nominal
        """
        xvals, shape, coefficients = self._prepare(*vals)

        if nominal:
            nominal_vals = self._eval(shape, coefficients)
            if errorband:
                if self._cov is None:
                    raise RuntimeError("Can only compute error band after covariance matrix loaded using update_from_roofit()")
                band = compute_band(self, shape, coefficients, nominal_vals)
                return nominal_vals, band
            else:
                return nominal_vals

        parameters = self.flat_parameters
        out = np.full(coefficients.shape[0], None)
        for i in range(coefficients.shape[0]):
            # sum small coefficients first
            order = np.argsort(coefficients[i])
            p = np.sum(parameters[order] * coefficients[i][order])
            dimstr = "_".join("%s%.3f" % (d, v[i]) for d, v in zip(self._dim_names, xvals))
            p.name = self.name + "_eval_" + dimstr.replace(".", "p")
            p.intermediate = False
            out[i] = p
        return out.reshape(shape)


class BernsteinPoly(BasisPoly):
    """
    Backcompatibility subclass of BasisPoly with fixed poly basis
    """

    def __init__(self, name, order, dim_names=None, init_params=None, limits=None, coefficient_transform=None):
        super(BernsteinPoly, self).__init__(name=name, order=order, dim_names=dim_names, basis="Bernstein", init_params=init_params, limits=limits, coefficient_transform=coefficient_transform)
        warnings.warn("BernsteinPoly is deprecated. Consider switching to BasisPoly(..., basis='Bernstein', ...)")


class ProductBasisPoly:
    """Composite function representing the product of multiple BasisPoly objects.

    Supports a subset of BasisPoly interface.

    Parameters:
        name: will be used to prefix any RooFit object names
        factors: list of BasisPoly objects that are multiplied together
    """
    def __init__(self, name, factors):
        if not factors:
            raise ValueError("ProductBasisPoly requires at least one BasisPoly factor")
        self._functions = list(factors)

        # deterministic parameter aggregation with deduplication
        param_dict = OrderedDict()
        dim_names = []
        for f in self._functions:
            if not isinstance(f, BasisPoly):
                raise TypeError("All factors must be BasisPoly instances")
            for p in f.parameters:
                if p.name not in param_dict:
                    param_dict[p.name] = p
            for d in f.dim_names:
                if d not in dim_names:
                    dim_names.append(d)
        self._params = np.array(list(param_dict.values()))
        self._dim_names = dim_names

    @property
    def name(self):
        return self._name

    @property
    def parameters(self):
        return self._params

    @property
    def flat_parameters(self):
        return self._params.reshape(-1)

    @property
    def dim_names(self):
        return self._dim_names

    def set_par_by_name(self, parname, parvalue):
        for f in self._functions:
            f.set_par_by_name(parname, parvalue)

    def update_from_roofit(self, fit_result):
        par_names = sorted([p for p in fit_result.floatParsFinal().contentsString().split(",") if any([p==par.name for par in self._params])])
        means, cov = params_from_roofit(fit_result, par_names)
        par_results = {p: round(means[i], 3) for i, p in enumerate(par_names)}
        for par in self.flat_parameters:
            par.value = par_results[par.name]
        self._cov = cov

        # call for each factor to update parameter values
        for f in self._functions:
            f.update_from_roofit(fit_result)

    def _eval(self, shape, coefficients):
        for ifn, f in enumerate(self._functions):
            tmp = f._eval(shape[ifn], coefficients[ifn])
            if ifn==0:
                results = tmp
            else:
                results *= tmp
        return results

    def __call__(self, *vals, nominal: bool = True, errorband: bool = False):
        if not nominal:
            raise ValueError("ProductBasisPoly currently does not implement nominal=False")

        prepared = []
        for f in self._functions:
            # get x values only for dimensions used in this function
            f_dim_indices = [self._dim_names.index(d) for d in f.dim_names]
            vals_for_f = [vals[i] for i in f_dim_indices]
            prepared.append(f._prepare(*vals_for_f))
        _, shape, coefficients = zip(*prepared)
        nominal_vals = self._eval(shape, coefficients)
        if errorband:
            if self._cov is None:
                raise RuntimeError("Can only compute error band after covariance matrix loaded using update_from_roofit()")
            band = compute_band(self, shape, coefficients, nominal_vals)
            return nominal_vals, band
        else:
            return nominal_vals


class DecorrelatedNuisanceVector:
    """A decorrelated nuisance vector

    This class is used to create a vector of nuisance parameters that are
    decorrelated from a set of correlated parameters and a covariance matrix.
    This is useful for creating a set of constrained nuisance parameters in combine,
    as combine does not support correlated nuisance parameters.

    Parameters:
        prefix: a prefix for the names of the parameters
        param_in: a numpy array of means for the nuisance parameters
        transform: a numpy array of coefficients, derived from the singular value decomposition of the covariance matrix of the nuisance parameters
    """

    def __init__(self, prefix: str, param_in: np.ndarray, transform: np.ndarray):
        if not isinstance(param_in, np.ndarray):
            raise ValueError("Expecting param_in to be numpy array")
        if not isinstance(transform, np.ndarray):
            raise ValueError("Expecting transform to be numpy array")
        if not (len(param_in.shape) == 1 and len(transform.shape) == 2 and transform.shape[0] == param_in.shape[0] and transform.shape[1] == param_in.shape[0]):
            raise ValueError("param_in and transform have mismatched shapes")

        self._transform = transform
        self._parameters = np.array([NuisanceParameter(prefix + str(i + 1), "param") for i in range(param_in.size)])
        self._correlated = np.full(self._parameters.shape, None)
        for i in range(self._parameters.size):
            coef = self._transform[:, i]
            order = np.argsort(np.abs(coef))
            self._correlated[i] = np.sum(coef[order] * self._parameters[order]) + param_in[i]

    @classmethod
    def fromRooFitResult(cls, prefix: str, fitresult, param_names: Optional[List[str]] = None):
        """Create a DecorrelatedNuisanceVector from a RooFit result

        Parameters:
            prefix: a prefix for the names of the parameters
            fitresult: a RooFitResult object
            param_names: optional list of parameter names to include in the vector. If None,
                all parameters in the fit result will be included.
        """
        means, cov = params_from_roofit(fitresult, param_names)
        transform = svd_transform(cov)
        out = cls(prefix, means, transform)
        if param_names is not None:
            for p, name in zip(out.correlated_params, param_names):
                p.name = name
        return out

    @property
    def parameters(self):
        return self._parameters

    @property
    def correlated_params(self):
        return self._correlated

    @property
    def correlated_str(self):
        return "\n".join(p.formula(rendering=True) for p in self._correlated)
