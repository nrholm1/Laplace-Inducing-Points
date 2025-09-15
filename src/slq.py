import jax
import jax.numpy as jnp

from matfree import decomp, stochtrace as matfree_stochtrace
from src.matfree_monkeypatch import integrand_funm_sym_logdet


def estimate_logdet_slq(Xfun, *, D: int, M: int, key, slq_samples: int, slq_num_matvecs: int | None):
    k = M if slq_num_matvecs is None else min(slq_num_matvecs, M)
    x0 = jnp.zeros((D,), dtype=jnp.float32)
    tridiag = decomp.tridiag_sym(k)
    problem = integrand_funm_sym_logdet(tridiag)
    sampler = matfree_stochtrace.sampler_rademacher(x0, num=slq_samples)
    estimator = matfree_stochtrace.estimator(problem, sampler=sampler)
    keys = jax.random.split(key, slq_samples)

    def one(kk):
        return estimator(Xfun, kk)

    logdets = jax.lax.map(jax.checkpoint(one), keys)
    return logdets.mean()