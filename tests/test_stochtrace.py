import jax
import jax.numpy as jnp
import pytest

from src.stochtrace import stochastic_trace_estimator_mvp, stochastic_trace_estimator_dense, na_hutchpp_inv_mvp, hutchpp_dense, na_hutchpp_dense, hutchpp_mvp, na_hutchpp_mvp, hutchpp_inv_mvp
from fixtures import matrix_test_suite

jax.config.update("jax_enable_x64", True)



def test_hutchinson_dense(matrix_test_suite):
    M1,M2,M3 = matrix_test_suite
    seed = jax.random.PRNGKey(seed=2894598)
    
    tr1_approx = stochastic_trace_estimator_dense(M1, seed, num_samples=3)
    tr1_exact  = jnp.trace(M1)
    assert jnp.isclose(tr1_approx, tr1_exact, rtol=1e-2), f"Error for M1. True:{tr1_exact:.2f}, Approx.:{tr1_approx:.2f}"
    
    tr2_approx = stochastic_trace_estimator_dense(M2, seed, num_samples=10)
    tr2_exact  = jnp.trace(M2)
    assert jnp.isclose(tr2_approx, tr2_exact, rtol=3e-2), f"Error for M2. True:{tr2_exact:.2f}, Approx.:{tr2_approx:.2f}"
    
    tr3_approx = stochastic_trace_estimator_dense(M3, seed, num_samples=10)
    tr3_exact  = jnp.trace(M3)
    assert jnp.isclose(tr3_approx, tr3_exact, rtol=3e-2), f"Error for M3. True:{tr3_exact:.2f}, Approx.:{tr3_approx:.2f}"


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
    




def test_hutchpp_dense(matrix_test_suite):
    M1,M2,M3 = matrix_test_suite
    seed = jax.random.PRNGKey(seed=284598)
    
    tr1_approx = hutchpp_dense(M1, seed, num_samples=3)
    tr1_exact  = jnp.trace(M1)
    assert jnp.isclose(tr1_approx, tr1_exact, rtol=1e-2),f"Error for M1. True:{tr1_exact:.2f}, Approx.:{tr1_approx:.2f}"
    
    tr2_approx = hutchpp_dense(M2, seed, num_samples=3)
    tr2_exact  = jnp.trace(M2)
    assert jnp.isclose(tr2_approx, tr2_exact, rtol=1e-2), f"Error for M2. True:{tr2_exact:.2f}, Approx.:{tr2_approx:.2f}"
    
    tr3_approx = hutchpp_dense(M3, seed, num_samples=10)
    tr3_exact  = jnp.trace(M3)
    assert jnp.isclose(tr3_approx, tr3_exact, rtol=1e-2), f"Error for M3. True:{tr3_exact:.2f}, Approx.:{tr3_approx:.2f}"
    

def test_hutchpp_mvp(matrix_test_suite):
    M1,M2,M3 = matrix_test_suite
    seed = jax.random.PRNGKey(seed=284598)
    
    
    def M1fun(v): return M1@v
    tr1_approx = hutchpp_mvp(M1fun, D=M1.shape[0], seed=seed, num_samples=3)
    tr1_exact  = jnp.trace(M1)
    assert jnp.isclose(tr1_approx, tr1_exact, rtol=1e-2), f"Error for M1. True:{tr1_exact:.2f}, Approx.:{tr1_approx:.2f}"
    
    def M2fun(v): return M2@v
    tr2_approx = hutchpp_mvp(M2fun, M2.shape[0], seed, num_samples=3)
    tr2_exact  = jnp.trace(M2)
    assert jnp.isclose(tr2_approx, tr2_exact, rtol=1e-2), f"Error for M2. True:{tr2_exact:.2f}, Approx.:{tr2_approx:.2f}"
    
    def M3fun(v): return M3@v
    tr3_approx = hutchpp_mvp(M3fun, M3.shape[0], seed, num_samples=10)
    tr3_exact  = jnp.trace(M3)
    assert jnp.isclose(tr3_approx, tr3_exact, rtol=1e-2), f"Error for M3. True:{tr3_exact:.2f}, Approx.:{tr3_approx:.2f}"
    

def test_na_hutchpp_dense(matrix_test_suite):
    M1,M2,M3 = matrix_test_suite
    seed = jax.random.PRNGKey(seed=2894598)
    
    tr1_approx = na_hutchpp_dense(M1, seed, num_samples=3)
    tr1_exact  = jnp.trace(M1)
    assert jnp.isclose(tr1_approx, tr1_exact, rtol=1e-2),f"Error for M1. True:{tr1_exact:.2f}, Approx.:{tr1_approx:.2f}"
    
    tr2_approx = na_hutchpp_dense(M2, seed, num_samples=3)
    tr2_exact  = jnp.trace(M2)
    assert jnp.isclose(tr2_approx, tr2_exact, rtol=1e-2), f"Error for M2. True:{tr2_exact:.2f}, Approx.:{tr2_approx:.2f}"
    
    tr3_approx = na_hutchpp_dense(M3, seed, num_samples=40)
    tr3_exact  = jnp.trace(M3)
    assert jnp.isclose(tr3_approx, tr3_exact, rtol=1e-2), f"Error for M3. True:{tr3_exact:.2f}, Approx.:{tr3_approx:.2f}"


def test_na_hutchpp_mvp(matrix_test_suite):
    M1,M2,M3 = matrix_test_suite
    seed = jax.random.PRNGKey(seed=2894598)
    
    def M1fun(v): return M1@v
    tr1_approx = na_hutchpp_mvp(M1fun, M1.shape[0],seed, num_samples=3)
    tr1_exact  = jnp.trace(M1)
    assert jnp.isclose(tr1_approx, tr1_exact, rtol=1e-2),f"Error for M1. True:{tr1_exact:.2f}, Approx.:{tr1_approx:.2f}"
    
    def M2fun(v): return M2@v
    tr2_approx = na_hutchpp_mvp(M2fun, M2.shape[0], seed, num_samples=3)
    tr2_exact  = jnp.trace(M2)
    assert jnp.isclose(tr2_approx, tr2_exact, rtol=1e-2), f"Error for M2. True:{tr2_exact:.2f}, Approx.:{tr2_approx:.2f}"
    
    def M3fun(v): return M3@v
    tr3_approx = na_hutchpp_mvp(M3fun, M3.shape[0],seed, num_samples=40)
    tr3_exact  = jnp.trace(M3)
    assert jnp.isclose(tr3_approx, tr3_exact, rtol=1e-2), f"Error for M3. True:{tr3_exact:.2f}, Approx.:{tr3_approx:.2f}"
    

def test_inv_hutchpp(matrix_test_suite):
    M1,M2,M3 = matrix_test_suite
    M3 = M3[:2500, :2500] # slice for pinv!
    seed = jax.random.PRNGKey(seed=2894598)
    
    def M1fun(v): 
        return M1@v
    tr1_inv_approx = hutchpp_inv_mvp(M1fun, M1.shape[0], seed, num_samples=3)
    tr1_inv_exact  = jnp.trace(jnp.linalg.pinv(M1))
    assert jnp.isclose(tr1_inv_approx, tr1_inv_exact, rtol=1e-2), f"Error for M1. True:{tr1_inv_exact:.2f}, Approx.:{tr1_inv_approx:.2f}"
    
    def M2fun(v): 
        return M2@v
    tr2_inv_approx = hutchpp_inv_mvp(M2fun, M2.shape[0], seed, num_samples=3)
    tr2_inv_exact  = jnp.trace(jnp.linalg.pinv(M2))
    assert jnp.isclose(tr2_inv_approx, tr2_inv_exact, rtol=1e-2), f"Error for M2. True:{tr2_inv_exact:.2f}, Approx.:{tr2_inv_approx:.2f}"
    
    def M3fun(v): 
        return M3@v
    tr3_inv_approx = hutchpp_inv_mvp(M3fun, M3.shape[0], seed, num_samples=40)
    tr3_inv_exact  = jnp.trace(jnp.linalg.pinv(M3))
    assert jnp.isclose(tr3_inv_approx, tr3_inv_exact, rtol=1e-2), f"Error for M3. True:{tr3_inv_exact:.2f}, Approx.:{tr3_inv_approx:.2f}"
    
    
def test_inv_na_hutchpp(matrix_test_suite):
    M1,M2,M3 = matrix_test_suite
    M3 = M3[:2500, :2500] # slice for pinv!
    seed = jax.random.PRNGKey(seed=2894598)
    
    def M1fun(v): 
        return M1@v
    tr1_inv_approx = na_hutchpp_inv_mvp(M1fun, M1.shape[0], seed, num_samples=3)
    tr1_inv_exact  = jnp.trace(jnp.linalg.pinv(M1))
    assert jnp.isclose(tr1_inv_approx, tr1_inv_exact, rtol=1e-2), f"Error for M1. True:{tr1_inv_exact:.2f}, Approx.:{tr1_inv_approx:.2f}"
    
    def M2fun(v): 
        return M2@v
    tr2_inv_approx = na_hutchpp_inv_mvp(M2fun, M2.shape[0], seed, num_samples=3)
    tr2_inv_exact  = jnp.trace(jnp.linalg.pinv(M2))
    assert jnp.isclose(tr2_inv_approx, tr2_inv_exact, rtol=1e-2), f"Error for M2. True:{tr2_inv_exact:.2f}, Approx.:{tr2_inv_approx:.2f}"
    
    def M3fun(v): 
        return M3@v
    tr3_inv_approx = na_hutchpp_inv_mvp(M3fun, M3.shape[0], seed, num_samples=40)
    tr3_inv_exact  = jnp.trace(jnp.linalg.pinv(M3))
    assert jnp.isclose(tr3_inv_approx, tr3_inv_exact, rtol=1e-2), f"Error for M3. True:{tr3_inv_exact:.2f}, Approx.:{tr3_inv_approx:.2f}"