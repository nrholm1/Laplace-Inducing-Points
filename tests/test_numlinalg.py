import jax
import jax.numpy as jnp
import pytest

from src.ggn import stochastic_trace_estimator_full_2, stochastic_trace_estimator_jvp

jax.config.update("jax_enable_x64", True)

@pytest.fixture
def matrix_test_suite():
    """
    Create PSD matrices with varying spectrum magnitude for verifying numerical stability.
    PSD makes it less random since there otherwise might be some sign issues.
    Also, PSD will always be the case for GGN, by design.
    """
    # trace = 6
    M1 = jnp.diag(jnp.array([1.,2.,3.]))
    
    # trace = 10
    M2 = jnp.array([ 
        [  1., 4,  50],
        [-30,  4., 16],
        [ 12,  6,   5.],
    ])
    M2 = M2@M2.T
    
    M3 = jax.random.normal(key=jax.random.PRNGKey(seed=45895), shape=(500,500))
    M3 = M3@M3.T
    
    return M1,M2,M3



def test_stochtrace_full(matrix_test_suite):
    M1,M2,M3 = matrix_test_suite
    seed = jax.random.PRNGKey(seed=2894598)
    
    tr1_approx = stochastic_trace_estimator_full_2(M1, seed, num_samples=3)
    tr1_exact  = jnp.trace(M1)
    assert jnp.isclose(tr1_approx, tr1_exact, rtol=1e-2), "Error for M1"
    
    tr2_approx = stochastic_trace_estimator_full_2(M2, seed, num_samples=3)
    tr2_exact  = jnp.trace(M2)
    assert jnp.isclose(tr2_approx, tr2_exact, rtol=1e-2), "Error for M2"
    
    tr3_approx = stochastic_trace_estimator_full_2(M3, seed, num_samples=10)
    tr3_exact  = jnp.trace(M3)
    assert jnp.isclose(tr3_approx, tr3_exact, rtol=1e-2), "Error for M3"
    

def test_stochtrace_jvp(matrix_test_suite):
    M1,M2,M3 = matrix_test_suite
    seed = jax.random.PRNGKey(seed=2894598)
    
    
    M1fun = lambda v: M1@v
    tr1_approx = stochastic_trace_estimator_jvp(M1fun, M1.shape[0], seed, num_samples=3)
    tr1_exact  = jnp.trace(M1)
    assert jnp.isclose(tr1_approx, tr1_exact, rtol=1e-2), "Error for M1"
    
    M2fun = lambda v: M2@v
    tr2_approx = stochastic_trace_estimator_jvp(M2fun, M2.shape[0], seed, num_samples=3)
    tr2_exact  = jnp.trace(M2)
    assert jnp.isclose(tr2_approx, tr2_exact, rtol=1e-2), "Error for M2"
    
    M3fun = lambda v: M3@v
    tr3_approx = stochastic_trace_estimator_jvp(M3fun, M3.shape[0], seed, num_samples=10)
    tr3_exact  = jnp.trace(M3)
    assert jnp.isclose(tr3_approx, tr3_exact, rtol=1e-2), "Error for M3"
    