import numpy as np
import jax
import jax.numpy as jnp
import jax.flatten_util
import jax.profiler
import jax.tree_util

from matfree import decomp
from matfree.funm import funm_lanczos_sym, dense_funm_sym_eigh, funm_arnoldi

from src.lla import posterior_lla_dense, compute_curvature_approx_dense, predict_lla_dense, predict_lla_scalable
from src.sample import sample, inv_matsqrt_vp, sample_both, sample_dense, inv_matsqrt_dense
from src.ggn import compute_ggn_dense, compute_W_vps, build_WTW
from src.utils import flatten_nn_params, is_pd
from fixtures import regression_1d_data, small_model_state, classifier_state, classification_2d_data, sine_data, toyregressor_state


# Test #1: W linear operator
def test_WT_W_vps(regression_1d_data, small_model_state):
    """
    1) Compute W,WT linear operator oracle
    2) Verify W(WT(I)) = GGN
    """
    X, y = regression_1d_data
    state = small_model_state    
    flat_params, _ = flatten_nn_params(state.params)
    D = flat_params.shape[0]
    I = jnp.eye(D)
    
    full_GGN, *_ = compute_ggn_dense(state, X,  model_type="regressor")
    
    # WTfun = compute_WT_vp(state, X, model_type="regressor")
    # dummy_primal = jnp.ones(D)
    # Wfunh = jax.linear_transpose(WTfun, dummy_primal)
    # def Wfun(v): 
    #     (vp_res,) = Wfunh(v)
    #     return vp_res
    Wfun, WTfun = compute_W_vps(state, X, "regressor")
    WT_out = jax.vmap(WTfun, in_axes=1)(I)
    composite_GGN = jax.vmap(Wfun, in_axes=0)(WT_out)
    
    assert jnp.all(jnp.isclose(composite_GGN, full_GGN, atol=1e-8)), "GGNs don't match!"
    
    # test it as a single composite vp
    def composite_vp(v):
        return Wfun(WTfun(v))
    
    composite_GGN = jax.vmap(composite_vp, in_axes=0)(I)
    assert jnp.all(jnp.isclose(composite_GGN, full_GGN, atol=1e-8)), "GGNs don't match!"    

def test_WT_W_vps_2(classification_2d_data, classifier_state):
    """
    1) Compute W,WT linear operator oracle
    2) Verify W(WT(I)) = GGN
    """
    jax.config.update("jax_enable_x64", True) # 64 bit floats
    
    X, y = classification_2d_data
    X = X.astype(jnp.float64)
    y = y.astype(jnp.float64)
    
    state = classifier_state
    state = state.replace(
        params=jax.tree_util.tree_map(lambda param: param.astype(jnp.float64), state.params)
    )
    
    flat_params, _ = flatten_nn_params(state.params)
    D = flat_params.shape[0]
    I = jnp.eye(D)
    
    full_GGN, *_ = compute_ggn_dense(state, X,  model_type="classifier")
    
    # WTfun = compute_WT_vp(state, X, model_type="regressor")
    # dummy_primal = jnp.ones(D)
    # Wfunh = jax.linear_transpose(WTfun, dummy_primal)
    # def Wfun(v): 
    #     (vp_res,) = Wfunh(v)
    #     return vp_res
    Wfun, WTfun = compute_W_vps(state, X, "classifier")
    WT_out = jax.vmap(WTfun, in_axes=1)(I)
    composite_GGN = jax.vmap(Wfun, in_axes=0)(WT_out)
    
    assert jnp.all(jnp.isclose(composite_GGN, full_GGN, atol=1e-8)), "GGNs don't match!"
    
    # test it as a single composite vp
    def composite_vp(v):
        return Wfun(WTfun(v))
    
    composite_GGN = jax.vmap(composite_vp, in_axes=0)(I)
    assert jnp.all(jnp.isclose(composite_GGN, full_GGN, atol=1e-8)), "GGNs don't match!"
    
    alpha = 0.37
    alpha_inv = 1. / alpha
    
    dummy = WTfun(jnp.zeros(D))
    inner_shape = dummy.shape
    d           = dummy.size
    I_d         = jnp.eye(d, dtype=float)
    WTW = build_WTW(Wfun, WTfun, inner_shape, d, dtype=float, block=1)
    
    _,logdet_WTW = jnp.linalg.slogdet(I_d + alpha_inv*WTW)
    logdet_term = logdet_WTW + D*jnp.log(alpha) # ! drop last term since it does not matter for optimization
    trace_WTW = jnp.linalg.trace(WTW)
    trace_term = trace_WTW + D*alpha # ! drop last term since it does not matter for optimization
    pass



# Test #2: [v -> W(W^T@W)^{-1}W^T @ v] composite linear operator
def test_nullproj(sine_data, toyregressor_state):
    """
    1) Compute W,WT linear operator oracle
    2) Verify v -> W(
                    WTW_inv(
                            WT(v)
                        ) 
                    ) works.
    """
    jax.config.update("jax_enable_x64", True) # 64 bit floats
    
    train_loader, test_loader = sine_data
    X,y = next(iter(test_loader))
    
    # convert stuff to f64
    X = X.astype(jnp.float64)
    y = y.astype(jnp.float64)
    state = toyregressor_state
    state = state.replace(
        params=jax.tree_util.tree_map(lambda param: param.astype(jnp.float64), state.params)
    )
    
    flat_params, _ = flatten_nn_params(state.params)
    D = flat_params.shape[0]
    
    dummy_primal = jax.random.normal(key=jax.random.PRNGKey(41234), shape=(D,)) * 10
    Wfun, WTfun = compute_W_vps(state, X, "regressor")

    def composite_vp(v):
        return WTfun(Wfun(v))
    
    def composite_inv_vp(v):
        x,info = jax.scipy.sparse.linalg.cg(A=composite_vp, b=v)
        return x
    
    def nullproj_vp(v):
        return v - Wfun(composite_inv_vp(WTfun(v)))
    
    full_out = nullproj_vp(dummy_primal)
    
    assert full_out.shape == (D,), "Something is incorrect with the shapes"

    assert jnp.all(jnp.isclose(Wfun(WTfun(full_out)), jnp.zeros_like(full_out), atol=1.5e-3)), "full_out should be in kernel, i.e. GGN maps it to 0."



def test_nullproj_classifier(classification_2d_data, classifier_state):
    """
    1) Compute WT linear operator oracle
    2) Compute W oracle from WT oracle using jax.linear_transpose
    3) Verify v -> W(
                    WTW_inv(
                            WT(v)
                        ) 
                    ) works.
    """
    jax.config.update("jax_enable_x64", True) # 64 bit floats
    
    X, y = classification_2d_data
    X = X.astype(jnp.float64)
    y = y.astype(jnp.float64)
    state = classifier_state
    state = state.replace(
        params=jax.tree_util.tree_map(lambda param: param.astype(jnp.float64), state.params)
    )
    flat_params, _ = flatten_nn_params(state.params)
    D = flat_params.shape[0]
    key = jax.random.PRNGKey(1392)
    alpha = 0.5
    
    N = X.shape[0]
    
    dummy_primal = jax.random.normal(key=jax.random.PRNGKey(41234), shape=(D,))
    Wfun, WTfun = compute_W_vps(state, X, "classifier")

    def composite_vp(v):
        return WTfun(Wfun(v))
    
    def composite_inv_vp(v):
        x,info = jax.scipy.sparse.linalg.cg(A=composite_vp, b=v)
        return x
    
    def nullproj_vp(v):
        return v - Wfun(composite_inv_vp(WTfun(v)))
    
    full_out = nullproj_vp(dummy_primal)
    
    """
    BATCHED version below:
    """
    
    Wfun_b, WTfun_b = compute_W_vps(state, X, "classifier", blockwise=True)
    
    def composite_vp_b(i,v):
        return WTfun_b(i, Wfun_b(i, v))
    
    def composite_inv_vp_b(i,v):
        x,info = jax.scipy.sparse.linalg.cg(A=lambda v: composite_vp_b(i,v), b=v)
        return x
    
    def nullproj_fun_b(i,v):
        return v - Wfun_b(i, composite_inv_vp_b(i, WTfun_b(i,v)))
    
    # ? version that shuffles the minibatch order
    def nullproj_approx_shuffled(v, key, steps=5):
        def outer_body(step, state):
            v, key = state
            # generate a new permutation for each outer iteration
            key, subkey = jax.random.split(key)
            perm = jax.random.permutation(subkey, N)
            def inner_body(i, v):
                b = perm[i]
                return nullproj_fun_b(b, v)
            v = jax.lax.fori_loop(0, N, inner_body, v)
            return (v, key)
        v, _ = jax.lax.fori_loop(0, steps, outer_body, (v, key))
        return v
    
    def nullproj_approx(v, steps=5):
        def outer_body(step, v):
            def inner_body(b, v):
                return nullproj_fun_b(b, v)
            return jax.lax.fori_loop(0, N, inner_body, v)
        return jax.lax.fori_loop(0, steps, outer_body, v)
    # ! nonbatched (full) versions
    
    batched = nullproj_approx_shuffled(dummy_primal, key=key, steps=1_500) # ! ridiculous amount of steps
    projected = Wfun(WTfun(batched))

    assert jnp.all(jnp.isclose(projected, jnp.zeros_like(projected), atol=1e-8)), "full_out should be in kernel, i.e. GGN maps it to 0."


# Test #3: minibatched/streamed approach to compute
def test_minibatched_projection(sine_data, toyregressor_state):
    """
    1) Compute exact projection term
    2) Compute minibatched approximation
    3) Assert they are equal up to numerical error.
    """
    jax.config.update("jax_enable_x64", True) # 64 bit floats
    
    train_loader, test_loader = sine_data
    X,y = next(iter(train_loader))
    # X,y = next(iter(test_loader))
    N = X.shape[0]
    
    # convert stuff to f64
    X = X.astype(jnp.float64)
    y = y.astype(jnp.float64)
    state = toyregressor_state
    state = state.replace(
        params=jax.tree_util.tree_map(lambda param: param.astype(jnp.float64), state.params)
    )
    
    flat_params, _ = flatten_nn_params(state.params)
    D = flat_params.shape[0]
    
    dummy_primal = jnp.ones(D, dtype=jnp.float64)
    
    # ! batched versions
    # WTfun_b = compute_WT_vp(state, X, model_type="regressor", blockwise=True)
    # Wfunh_b = lambda i: jax.linear_transpose(lambda v: WTfun_b(i,v), dummy_primal)
    # def Wfun_b(i,v): 
    #     (vp_res,) = Wfunh_b(i)(v)
    #     return vp_res
    
    Wfun_b, WTfun_b = compute_W_vps(state, X, "regressor", blockwise=True)
    
    def composite_vp_b(i,v):
        return WTfun_b(i, Wfun_b(i, v))
    
    def composite_inv_vp_b(i,v):
        x,info = jax.scipy.sparse.linalg.cg(A=lambda v: composite_vp_b(i,v), b=v)
        return x
    
    def nullproj_fun_b(i,v):
        return v - Wfun_b(i, composite_inv_vp_b(i, WTfun_b(i,v)))
    
    # ? version that shuffles the minibatch order
    def nullproj_approx_shuffled(v, key, steps=5):
        def outer_body(step, state):
            v, key = state
            # generate a new permutation for each outer iteration
            key, subkey = jax.random.split(key)
            perm = jax.random.permutation(subkey, N)
            def inner_body(i, v):
                b = perm[i]
                return nullproj_fun_b(b, v)
            v = jax.lax.fori_loop(0, N, inner_body, v)
            return (v, key)
        v, _ = jax.lax.fori_loop(0, steps, outer_body, (v, key))
        return v
    
    def nullproj_approx(v, steps=5):
        def outer_body(step, v):
            def inner_body(b, v):
                return nullproj_fun_b(b, v)
            return jax.lax.fori_loop(0, N, inner_body, v)
        return jax.lax.fori_loop(0, steps, outer_body, v)
    # ! nonbatched (full) versions
        
    Wfun, WTfun = compute_W_vps(state, X, "regressor")
    dummy_primal = jax.random.normal(key=jax.random.PRNGKey(41234), shape=(D,))
    
    # def composite_vp(v):
    #     return WTfun(Wfun(v))
    
    # def composite_inv_vp(v):
    #     x,info = jax.scipy.sparse.linalg.cg(A=composite_vp, b=v)
    #     return x
    
    # def nullproj_vp(v):
    #     return v - Wfun(composite_inv_vp(WTfun(v)))
    
    # full = nullproj_vp(dummy_primal)
    # batched_s = nullproj_approx_shuffled(dummy_primal, key=jax.random.PRNGKey(1234), steps=100)
    batched = nullproj_approx(dummy_primal, steps=150_000) # ! ridiculous amount of steps
    projected = Wfun(WTfun(batched))
    # zeros = jnp.zeros_like(batched)
    
    # assert jnp.all( jnp.isclose(projected, 0., atol=1.5e-1) ), "Not all projected components deviate < 0.15 from 0."
    assert jnp.linalg.norm(projected, ord=2) <= 0.6 # E[||dummy_primal||_2] ≈ 15.5


def test_matfree_invsqrt():
    jax.config.update("jax_enable_x64", True) # 64 bit floats
    
    invsqrt_fun = dense_funm_sym_eigh(lambda x: 1.0/jnp.sqrt(x))
    tridiag = decomp.tridiag_sym(20)  # todo maybe set num_matvecs according to sample size - i.e. number of inducing points
    invmatsqrt = funm_lanczos_sym(invsqrt_fun, tridiag)
    
    D = 100
    matdiag = jnp.arange(1, D+1, dtype=jnp.float64) / D
    mat = jnp.diag(matdiag)
    def matvec(v):
        return mat @ v
    
    invsqrtmat = jnp.diag(1.0/jnp.sqrt(matdiag))
    def direct(v):
        return invsqrtmat @ v
    
    ones = jnp.ones(D, dtype=jnp.float64)
    mf_res = invmatsqrt(matvec, ones)
    res = direct(ones)
    
    assert jnp.all(jnp.isclose(res, mf_res, rtol=1e-1)), "hmm"


def test_matfree_invsqrt_classifier(classification_2d_data, classifier_state):
    jax.config.update("jax_enable_x64", True) # 64 bit floats
    
    X, y = classification_2d_data
    state = classifier_state
    
    # convert stuff to f64
    X = X.astype(jnp.float64)
    y = y.astype(jnp.float64)
    state = state.replace(
        params=jax.tree_util.tree_map(lambda param: param.astype(jnp.float64), state.params)
    )
    
    flat_params, _ = flatten_nn_params(state.params)
    D = flat_params.shape[0]
      
    dummy_primal = jax.random.normal(shape=(D,), key=jax.random.PRNGKey(484), dtype=jnp.float64)
    
    Wfun, WTfun = compute_W_vps(state, X, "classifier")

    def composite_vp(u):
        return WTfun(Wfun(u))
    
    
    invsqrt_fun = dense_funm_sym_eigh(lambda x: 1.0/jnp.sqrt(x))
    tridiag = decomp.tridiag_sym(20)  # todo maybe set num_matvecs according to sample size - i.e. number of inducing points
    invmatsqrt = funm_lanczos_sym(invsqrt_fun, tridiag)
    
    alpha = 0.5
    beta  = 1
    Wshape = WTfun(dummy_primal).shape

    def invmatsqrt_term(V):
        """
        v has shape (p,).  We do:
        1) V = WTfun(v) -> shape (M,K)
        2) flatten to shape (M*K,)
        3) call 'invmatsqrt' in that flattened space
        4) reshape back to (M,K).
        """
        Vflat = V.reshape(-1)           # shape (M*K,)

        def inner_fun_flat(Uflat):
            # 1) unflatten U to shape (M,K)
            Umat = Uflat.reshape(*Wshape)
            # 2) apply operator in matrix form
            #    This is alpha*U + beta*(WTfun(Wfun(U))),
            # WTUmat = WTfun(Wfun(Umat))  # shape (M,K)
            WTUmat = composite_vp(Umat)  # shape (M,K)
            WTUmat_flat = WTUmat.reshape(-1)  # shape (M*K,)
            # 3) also flatten the "alpha * Umat" part:
            return alpha * Uflat + beta * WTUmat_flat

        # Now do the Lanczos-based call in the 1D space of length M*K
        result_flat = invmatsqrt(inner_fun_flat, Vflat)  # result is shape (M*K,)

        # Optionally reshape the result back to (M, K) if needed:
        result = result_flat.reshape(*Wshape)
        return result
    
    # u = WTfun(v)    => (200, 2)
    # w = innerfun(u) => (200, 2)
    
    res = invmatsqrt_term(WTfun(dummy_primal))
    
    # ggn,*_ = compute_ggn_dense(state, X, "classifier")
    # evals,evecs = jnp.linalg.eigh(ggn)
    # evals = jnp.clip(evals, 0, jnp.inf)
    # ggn_sqrt = evecs @ jnp.diag(jnp.sqrt(evals)) @ evecs.T
    
    assert False
    
    
# def test_invsqrt_ggn(regression_1d_data, small_model_state):
def test_invsqrt_ggn(sine_data, toyregressor_state):
    
    # X, y = regression_1d_data
    # state = small_model_state    
    # flat_params, _ = flatten_nn_params(state.params)
    # D = flat_params.shape[0]
    
    jax.config.update("jax_enable_x64", True) # 64 bit floats
    
    train_loader, test_loader = sine_data
    X,y = next(iter(test_loader))
    N = X.shape[0]
    
    # convert stuff to f64
    X = X.astype(jnp.float64)
    y = y.astype(jnp.float64)
    state = toyregressor_state
    state = state.replace(
        params=jax.tree_util.tree_map(lambda param: param.astype(jnp.float64), state.params)
    )
    
    flat_params, _ = flatten_nn_params(state.params)
    D = flat_params.shape[0]
    
    dummy_primal = jax.random.normal(shape=(D,), key=jax.random.PRNGKey(484), dtype=jnp.float64)
    
    
    inv_matsqrt_fun = inv_matsqrt_vp(state, X, D, alpha=0.5, model_type='regressor')
        
    full_out = inv_matsqrt_fun(dummy_primal)

    assert full_out.shape == (D,), "Something is incorrect with the shapes"
    assert False, "NOT FINISHED - Should not currently pass!"


def test_sample_fun_tiny(regression_1d_data, small_model_state):
    X, y = regression_1d_data
    state = small_model_state    
    flat_params, _ = flatten_nn_params(state.params)
    D = flat_params.shape[0]
    key = jax.random.PRNGKey(1392)
    alpha = 0.5
    
    post_dist = posterior_lla_dense(small_model_state, X, alpha=1.0/alpha**0.5, model_type="regressor")
    samples = sample(state, X, D, alpha=alpha, key=key, model_type="regressor", num_samples=1_000)
    
    assert jnp.all(jnp.isclose(post_dist.mean(), samples.mean(axis=0), atol=1.1e-1)), "Means are not close!"
    assert jnp.all(jnp.isclose(post_dist.stddev(), samples.std(axis=0), atol=1e-1)),  "Stdevs are not close!"


def test_sample_fun_regressor(sine_data, toyregressor_state):
    jax.config.update("jax_enable_x64", True) # 64 bit floats
    
    train_loader, test_loader = sine_data
    X,y = next(iter(test_loader))
    N = X.shape[0]
    
    # convert stuff to f64
    X = X.astype(jnp.float64)
    y = y.astype(jnp.float64)
    state = toyregressor_state
    state = state.replace(
        params=jax.tree_util.tree_map(lambda param: param.astype(jnp.float64), state.params)
    )
    key = jax.random.PRNGKey(1392)
    alpha = 0.5
    
    flat_params, _ = flatten_nn_params(state.params)
    D = flat_params.shape[0]
    
    post_dist = posterior_lla_dense(state, X, model_type="regressor", alpha=alpha)
    samples = sample(state, X, D, alpha=alpha, key=key, model_type="regressor", num_samples=1_500)
    
    # todo try the Chi2 test from Søren here
    
    assert jnp.all(jnp.isclose(post_dist.mean(), samples.mean(axis=0), atol=1.1e-1)), "Means are not close!"
    assert jnp.all(jnp.isclose(post_dist.stddev(), samples.std(axis=0), atol=1e-1)),  "Stdevs are not close!"


def test_sample_fun_regressor_dense(sine_data, toyregressor_state):
    jax.config.update("jax_enable_x64", True) # 64 bit floats
    
    train_loader, test_loader = sine_data
    X,y = next(iter(test_loader))
    N = X.shape[0]
    
    # convert stuff to f64
    X = X.astype(jnp.float64)
    y = y.astype(jnp.float64)
    state = toyregressor_state
    state = state.replace(
        params=jax.tree_util.tree_map(lambda param: param.astype(jnp.float64), state.params)
    )
    key = jax.random.PRNGKey(1392)
    alpha = 0.5
    
    flat_params, _ = flatten_nn_params(state.params)
    D = flat_params.shape[0]
    I_D = jnp.eye(D)
    
    post_dist = posterior_lla_dense(state, X, model_type="regressor", alpha=alpha)
    # samples = sample_dense(state, X, D, alpha=alpha, key=key, model_type="regressor", num_samples=1_500)
    samples, dense_samples = sample_both(state, X, D, alpha=alpha, key=key, model_type="regressor", num_samples=500)
    
    # ggn,*_ = compute_ggn_dense(state, X, "regressor")
    # invmatsqrt_ggn = inv_matsqrt_dense(state, X, D, alpha, model_type="regressor")
    
    pass


def test_sample_fun_classifier(classification_2d_data, classifier_state):
    jax.config.update("jax_enable_x64", True) # 64 bit floats
    
    X, y = classification_2d_data
    X = X.astype(jnp.float64)
    y = y.astype(jnp.float64)
    
    Xnew = jax.random.normal(key=jax.random.PRNGKey(1), shape=(3, *X[0].shape))
    
    state = classifier_state
    state = state.replace(
        params=jax.tree_util.tree_map(lambda param: param.astype(jnp.float64), state.params)
    )
    flat_params,unravel_fn = flatten_nn_params(state.params['params'])
    D = flat_params.shape[0]
    key = jax.random.PRNGKey(1392)
    alpha = 0.5
    
    post_dist = posterior_lla_dense(state, X, model_type="classifier", alpha=alpha)
    samples = sample(state, X, D, alpha=alpha, key=key, model_type="classifier", num_samples=100)
    
    fdist = predict_lla_dense(state, Xnew, X, model_type="classifier", alpha=alpha)
    fmu, fcov = predict_lla_scalable(state, Xnew, X, model_type="classifier", alpha=alpha)
    
    assert False
    
def test_sample_fun_classifier_dense(classification_2d_data, classifier_state):
    jax.config.update("jax_enable_x64", True) # 64 bit floats
    
    X, y = classification_2d_data
    X = X.astype(jnp.float64)
    y = y.astype(jnp.float64)
    
    state = classifier_state
    state = state.replace(
        params=jax.tree_util.tree_map(lambda param: param.astype(jnp.float64), state.params)
    )
    key = jax.random.PRNGKey(1392)
    alpha = 0.5
    
    flat_params, _ = flatten_nn_params(state.params)
    D = flat_params.shape[0]
    I_D = jnp.eye(D)
    
    post_dist = posterior_lla_dense(state, X, model_type="classifier", alpha=alpha)
    # samples_dense = sample_dense(state, X, D, alpha=alpha, key=key, model_type="classifier", num_samples=10)
    # samples = sample(state, X, D, alpha=alpha, key=key, model_type="classifier", num_samples=10)
    samples, dense_samples = sample_both(state, X, D, alpha=alpha, key=key, model_type="classifier", num_samples=10)
    
    pass