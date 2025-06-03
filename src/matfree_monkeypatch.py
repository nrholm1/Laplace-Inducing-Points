from matfree.backend import func, linalg, np
from matfree.backend.typing import Callable
from matfree.funm import integrand_funm_sym

from jax.numpy import clip


def dense_funm_sym_eigh(matfun):
    """**[MONKEY-PATCHED!]**
    
    Implement dense matrix-functions via symmetric eigendecompositions.

    Use it to construct one of the matrix-free matrix-function implementations,
    e.g. [matfree.funm.funm_lanczos_sym][matfree.funm.funm_lanczos_sym].
    """

    def fun(dense_matrix):
        eigvals, eigvecs = linalg.eigh(dense_matrix)
        eigvals = clip(eigvals, min=1.0) # ! clip eigenvalues to 1 to become 0 in log!
        fx_eigvals = func.vmap(matfun)(eigvals)
        return eigvecs @ linalg.diagonal(fx_eigvals) @ eigvecs.T
    return fun


def integrand_funm_sym_logdet(tridiag_sym: Callable, /):
    """**[MONKEY-PATCHED!]**
    
    Construct the integrand for the log-determinant.

    This function assumes a symmetric, positive definite matrix.

    Parameters
    ----------
    tridiag_sym
        An implementation of tridiagonalisation.
        E.g., the output of
        [decomp.tridiag_sym][matfree.decomp.tridiag_sym].

    """
    dense_funm = dense_funm_sym_eigh(np.log)
    return integrand_funm_sym(dense_funm, tridiag_sym)