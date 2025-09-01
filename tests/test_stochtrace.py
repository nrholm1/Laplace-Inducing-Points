import jax
import jax.numpy as jnp
import pytest

from src.stochtrace import hutchpp, stochastic_trace_estimator_mvp
from fixtures import matrix_test_suite

jax.config.update("jax_enable_x64", True)


def test_hutchinson_mvp(matrix_test_suite):
    M1,M2,M3 = matrix_test_suite
    seed = jax.random.PRNGKey(seed=2894598)
    
    def M1fun(v): return M1@v
    tr1_approx = stochastic_trace_estimator_mvp(M1fun, M1.shape[0], seed, num_samples=3)
    tr1_exact  = jnp.trace(M1)
    assert jnp.isclose(tr1_approx, tr1_exact, rtol=1e-2), f"Error for M1. True:{tr1_exact:.2f}, Approx.:{tr1_approx:.2f}"
    
    def M2fun(v): return M2@v
    tr2_approx = stochastic_trace_estimator_mvp(M2fun, M2.shape[0], seed, num_samples=10)
    tr2_exact  = jnp.trace(M2)
    assert jnp.isclose(tr2_approx, tr2_exact, rtol=3e-2), f"Error for M2. True:{tr2_exact:.2f}, Approx.:{tr2_approx:.2f}"
    
    def M3fun(v): return M3@v
    tr3_approx = stochastic_trace_estimator_mvp(M3fun, M3.shape[0],seed, num_samples=10)
    tr3_exact  = jnp.trace(M3)
    assert jnp.isclose(tr3_approx, tr3_exact, rtol=3e-2), f"Error for M3. True:{tr3_exact:.2f}, Approx.:{tr3_approx:.2f}"
    


def test_hutchpp_mvp(matrix_test_suite):
    _,_,M3 = matrix_test_suite
    seed = jax.random.PRNGKey(seed=284598)

    def M3fun(v): return M3@v
    tr3_exact  = jnp.trace(M3)
    
    k = 3200
    eps = jax.random.rademacher(key=seed, shape=(k, M3.shape[0]))
    s2 = 32
    s1 = k - s2
    st_sampler = lambda _: eps
    stoch_trace = lambda vp: hutchpp(vp, st_sampler, s1=s1, s2=s2)
    trace_term = stoch_trace(M3fun)
    assert jnp.isclose(trace_term, tr3_exact, rtol=1e-8), f"Error for M3. True:{tr3_exact:.2f}, Approx.:{trace_term:.2f}"