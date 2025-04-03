import jax
import jax.numpy as jnp

from matfree import decomp, funm, stochtrace

from src.utils import count_model_params
from src.lla import compute_curvature_approx_dense, compute_curvature_approx
from src.stochtrace import hutchpp_mvp, na_hutchpp_mvp
from tests.fixtures import classification_2d_data, classifier_state

# jax.config.update("jax_enable_x64", True)


def compute_dense_terms(params, x, state, alpha, model_type, full_set_size=None):
    xind= params
    
    prior_std = alpha**(-0.5) # σ = 1/sqrt(⍺) = ⍺^(-1/2)
    # w_fake = jnp.ones_like(dataset[0])
    S_full_inv, *_ = compute_curvature_approx_dense(state, x, prior_std=prior_std, model_type=model_type, full_set_size=full_set_size, return_Hinv=True)
    S_induc,    *_ = compute_curvature_approx_dense(state, xind, prior_std=prior_std, model_type=model_type, full_set_size=full_set_size, return_Hinv=False)
    
    trace_term = jnp.linalg.trace(S_full_inv @ S_induc)
    # sign_full, logdet_full = jnp.linalg.slogdet(S_full_inv)
    sign_induc, logdet_induc_inv = jnp.linalg.slogdet(S_induc)
    # todo use signs to signal if determinants are nonpositive - does not play well with JIT
    logdet_term = - logdet_induc_inv # ! since inverse was used
    return trace_term, logdet_term


def scalable_trace_term(params, x, state, alpha, model_type, seed, full_set_size=None, trace_estimator=hutchpp_mvp):
    xind, w = params
    prior_std = alpha**(-0.5) # σ = 1/sqrt(⍺) = ⍺^(-1/2)
    
    D = count_model_params(state.params)
    
    # compute matrix free linear operator oracles
    S_full_vp = compute_curvature_approx(state, x, prior_std=prior_std, model_type=model_type, full_set_size=full_set_size)
    S_induc_vp = compute_curvature_approx(state, xind, prior_std=prior_std, model_type=model_type, full_set_size=full_set_size)
    
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


def scalable_logdet_term(params, x, state, alpha, model_type, seed, full_set_size=None, trace_estimator=hutchpp_mvp):
    xind = params
    prior_std = alpha**(-0.5) # σ = 1/sqrt(⍺) = ⍺^(-1/2)
    
    D = count_model_params(state.params)
    
    # compute matrix free linear operator oracles
    # S_full_vp = compute_curvature_approx(state, x, prior_std=prior_std, w=w_fake, model_type=model_type, full_set_size=full_set_size)
    S_induc_vp = compute_curvature_approx(state, xind, prior_std=prior_std, model_type=model_type, full_set_size=full_set_size)
    
    def stoch_lanczos_quadrature(Xfun):
        # adapted directly from 
        # https://pnkraemer.github.io/matfree/Tutorials/1_compute_log_determinants_with_stochastic_lanczos_quadrature/
        num_matvecs = 3
        tridiag_sym = decomp.tridiag_sym(num_matvecs)
        problem = funm.integrand_funm_sym_logdet(tridiag_sym)
        x_like = jnp.ones((D,), dtype=float)
        sampler = stochtrace.sampler_normal(x_like, num=150)
        estimator = stochtrace.estimator(problem, sampler=sampler)
        logdet = estimator(Xfun, seed)
        return logdet
    
    logdet_term = stoch_lanczos_quadrature(S_induc_vp)
    
    return logdet_term


def test_scalable_trace_term(classification_2d_data, classifier_state):
    key = jax.random.PRNGKey(seed=178189)
    induc_key, hutch_key = jax.random.split(key, 2)
    
    X,y = classification_2d_data
    N = X.shape[0] # full_set_size
    M = 10 # number of inducing points
    w = jnp.array(2.05) # dummy value
    Z = jax.random.uniform(key=induc_key, shape=(M, X.shape[1]))
    alpha = 0.5
    
    trace_dense,_ = compute_dense_terms(params=(Z,w),
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
    

def test_scalable_logdet_term(classification_2d_data, classifier_state):
    key = jax.random.PRNGKey(seed=178189)
    induc_key, hutch_key = jax.random.split(key, 2)
    
    X,y = classification_2d_data
    N = X.shape[0] # full_set_size
    M = 10 # number of inducing points
    w = jnp.array(2.05) # dummy value
    Z = jax.random.uniform(key=induc_key, shape=(M, X.shape[1]))
    alpha = 0.5
    
    _,logdet_dense = compute_dense_terms(params=Z,
                                   x=X, 
                                   state=classifier_state, 
                                   alpha=alpha, 
                                   model_type="classifier", 
                                   full_set_size=N)
    
    logdet_scalable = scalable_logdet_term(params=Z, 
                                         x=X, 
                                         state=classifier_state, 
                                         alpha=alpha, 
                                         seed=hutch_key, 
                                         model_type="classifier",
                                         full_set_size=N,
                                         trace_estimator=hutchpp_mvp)
    assert jnp.isclose(logdet_dense, logdet_scalable, rtol=1e-2), "MF logdet (hutchpp) does not match dense!"
    
    logdet_scalable_2 = scalable_logdet_term(params=Z, 
                                         x=X, 
                                         state=classifier_state, 
                                         alpha=alpha, 
                                         seed=hutch_key, 
                                         model_type="classifier",
                                         full_set_size=N,
                                         trace_estimator=na_hutchpp_mvp)
    assert jnp.isclose(logdet_dense, logdet_scalable_2, rtol=1e-2), "MF logdet (na-hutchpp) does not match dense!"