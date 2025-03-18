import jax
import jax.numpy as jnp

from src.utils import count_model_params
from src.lla import compute_curvature_approx_dense, compute_curvature_approx
from src.stochtrace import hutchpp_mvp, na_hutchpp_mvp
from tests.fixtures import classification_2d_data, classifier_state

# jax.config.update("jax_enable_x64", True)


def dense_trace_term(params, x, state, alpha, model_type, full_set_size=None):
    xind, w = params
    
    prior_std = alpha**(-0.5) # σ = 1/sqrt(⍺) = ⍺^(-1/2)
    # w_fake = jnp.ones_like(dataset[0])
    w_fake = jnp.array(1.)
    S_full_inv, *_ = compute_curvature_approx_dense(state, x, prior_std=prior_std, w=w_fake, model_type=model_type, full_set_size=full_set_size, return_Hinv=True)
    S_induc,    *_ = compute_curvature_approx_dense(state, xind, prior_std=prior_std, w=w, model_type=model_type, full_set_size=full_set_size, return_Hinv=False)
    
    trace_term = jnp.linalg.trace(S_full_inv @ S_induc)
    return trace_term


def scalable_trace_term(params, x, state, alpha, model_type, seed, full_set_size=None, trace_estimator=hutchpp_mvp):
    xind, w = params
    prior_std = alpha**(-0.5) # σ = 1/sqrt(⍺) = ⍺^(-1/2)
    w_fake = jnp.array(1.)
    
    D = count_model_params(state.params)
    
    # compute matrix free linear operator oracles
    S_full_vp = compute_curvature_approx(state, x, prior_std=prior_std, w=w_fake, model_type=model_type, full_set_size=full_set_size)
    S_induc_vp = compute_curvature_approx(state, xind, prior_std=prior_std, w=w, model_type=model_type, full_set_size=full_set_size)
    
    # ! option 1: use conjugate gradient
    @jax.jit
    def S_induc_inv_vp(v):
        x,info = jax.scipy.sparse.linalg.cg(A=S_induc_vp, b=v)
        return x
    
    @jax.jit
    def composite_vp(v):
        # computes S_Z^{-1} @ S_full
        # return S_full_vp(S_induc_inv_vp(v))
        return jax.vmap(S_full_vp, in_axes=1, out_axes=1)(
            jax.vmap(S_induc_inv_vp, in_axes=1, out_axes=1)(v)
        )
    
    trace_term = trace_estimator(composite_vp, D=D, seed=seed, num_samples=150)
    return trace_term


def test_scalable_trace_term(classification_2d_data, classifier_state):
    key = jax.random.PRNGKey(seed=178189)
    induc_key, hutch_key = jax.random.split(key, 2)
    
    X,y = classification_2d_data
    N = X.shape[0] # full_set_size
    M = 10 # number of inducing points
    w = jnp.array(2.05) # dummy value
    Z = jax.random.uniform(key=induc_key, shape=(M, X.shape[1]))
    alpha = 0.5
    
    trace_dense = dense_trace_term(params=(Z,w),
                                   x=X, 
                                   state=classifier_state, 
                                   alpha=alpha, 
                                   model_type="classifier", 
                                   full_set_size=N)
    
    trace_scalable = scalable_trace_term(params=(Z,w), 
                                         x=X, 
                                         state=classifier_state, 
                                         alpha=alpha, 
                                         seed=hutch_key, 
                                         model_type="classifier",
                                         full_set_size=N,
                                         trace_estimator=hutchpp_mvp)
    assert jnp.isclose(trace_dense, trace_scalable, rtol=1e-2), "MF trace (hutchpp) does not match dense!"
    
    trace_scalable_2 = scalable_trace_term(params=(Z,w), 
                                         x=X, 
                                         state=classifier_state, 
                                         alpha=alpha, 
                                         seed=hutch_key, 
                                         model_type="classifier",
                                         full_set_size=N,
                                         trace_estimator=na_hutchpp_mvp)
    assert jnp.isclose(trace_dense, trace_scalable_2, rtol=1e-2), "MF trace (na-hutchpp) does not match dense!"
    
    

def scalable_logdet_term(classification_2d_data, classifier_state):
    ... # todo
