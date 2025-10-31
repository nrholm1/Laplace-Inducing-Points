
# ! Everything in this file is hardcoded to classifiers!

#%%
import jax
import jax.numpy as jnp


def H_dense(p):
    """Closed-form loss Hessian of softmax."""
    return jnp.diag(p) - jnp.outer(p, p)
    
def Hsqrt_dense(p):
    """Non-Symmetric matrix S s.t. H = S Sᵀ"""
    q = jnp.sqrt(p)
    P = jnp.eye(p.size, dtype=p.dtype) - jnp.outer(q, q)
    S = q[None,:] * P
    return S

def Hsqrtsym_dense(p, tol=1e-6):
    """Symmetric matrix S s.t. H = S Sᵀ"""
    _H = H_dense(p)
    E,V = jnp.linalg.eigh(_H)
    S = jnp.where(E > tol, jnp.sqrt(E), 0.0)
    return (V*S) @ V.T


def HTsqrt_vp(p, u):
    s = jnp.sqrt(p)
    return s * u - (jnp.dot(s, u)) * p

def Hsqrt_vp(p, u):
    s = jnp.sqrt(p)
    return s * u - (jnp.dot(p, u)) * s

def H_vp(p, u):
    H = jnp.diag(p) - jnp.outer(p, p)
    return H @ u


def get_f_apply(apply_fn, unravel_fn):
    def f_apply(flatp, x):
        return apply_fn({"params": unravel_fn(flatp)}, x)
    return f_apply


def compute_ggn_dense(data, flat_params, apply_fn, unravel_fn, *, mode:str="fast"):
    f_apply_h = get_f_apply(apply_fn, unravel_fn)
    def f_apply(*args):
        y = f_apply_h(*args)
        return y,y
    
    def ggn_per_datum(xi):
        J, f_out = jax.jacrev(lambda flatp: f_apply(flatp, xi), has_aux=True)(flat_params)
        # J = J.squeeze() # ! for CNN only
        probs = jax.nn.softmax(f_out)
        H = H_dense(probs)
        return J.T @ H @ J
    
    def body(acc, xi):
        acc += ggn_per_datum(xi)
        return acc, None
    
    if mode == "fast":
        return jax.vmap(ggn_per_datum)(data).sum(axis=0)
    elif mode == "memeff":
        zeros = jnp.zeros((flat_params.shape[0],flat_params.shape[0]))
        return jax.lax.scan(body, zeros, data)[0]
    else:
        raise ValueError(f"unknown mode {mode}")



def compute_ggn_vp(data, flat_params, apply_fn, unravel_fn, *, mode:str="fast"):
    f_apply = get_f_apply(apply_fn, unravel_fn)
    
    def ggn_vp_per_datum(xi, v):
        f = lambda flatp: f_apply(flatp, xi)#.squeeze(0) # ! squeeze for CNN only
        
        f_out, Jv = jax.jvp(f, (flat_params,), (v,))
        _, JT_fun = jax.vjp(f, flat_params)
        probs = jax.nn.softmax(f_out)
        HJv = H_vp(probs, Jv)
        
        return JT_fun(HJv)[0]
    
    def body(acc, xi, v):
        acc += ggn_vp_per_datum(xi, v)
        return acc, None
    
    def ggn_vp(v):
        if mode == "fast":
            return jax.vmap(lambda datum: ggn_vp_per_datum(datum, v))(data).sum(axis=0)
        elif mode == "memeff":
            zeros = jnp.zeros_like(flat_params)
            return jax.lax.scan(lambda acc, datum: body(acc, datum, v), zeros, data)[0]
        else:
            raise ValueError(f"unknown mode {mode}")
    
    return ggn_vp
    

def compute_W_vps(data, flat_params, apply_fn, unravel_fn, *, mode:str="fast"):
    f_apply = get_f_apply(apply_fn, unravel_fn)
    
    def WT_per_datum(xi, v):
        f = lambda flatp: f_apply(flatp, xi)#.squeeze(0) # ! squeeze for CNN only
        
        f_out, Jv = jax.jvp(f, (flat_params,), (v,))
        probs = jax.nn.softmax(f_out)
        
        return Hsqrt_vp(probs, Jv)
    
    if mode == "fast":
        def WT_vp(v):
            return jax.vmap(lambda datum: WT_per_datum(datum, v))(data)
        W_vp_h = jax.linear_transpose(WT_vp, jnp.zeros_like(flat_params))
    elif mode == "memeff":
        def WT_vp(v):
            def body(carry, xi):
                yi = WT_per_datum(xi, v)
                return carry, yi
            _, ys = jax.lax.scan(body, None, data)
            return ys
        _, W_vp_h = jax.vjp(WT_vp, jnp.zeros_like(flat_params)) #! SLOWER than linear_transpose, but actually works for the scan approach here...
    else:
        raise ValueError(f"unknown mode {mode}")
    
    W_vp = lambda u: W_vp_h(u)[0]
    
    return W_vp, WT_vp



if __name__ == "__main__":
    key = jax.random.PRNGKey(123)
    key_logits, key_v = jax.random.split(key, 2)
    
    d = 5
    logits = jax.random.normal(key, (d,))
    probs = jax.nn.softmax(logits)
    v = jax.random.normal(key_v, (d,))
    
    Hd = H_dense(probs)
    Hvd = Hd @ v
    Hv = H_vp(probs, v)
    
    assert jnp.all(jnp.isclose(Hvd, Hv, atol=1e-6))
    
    
    Hsqrtd = Hsqrt_dense(probs)
    Hsqrtvd = Hsqrtd @ v
    Hsqrtv = Hsqrt_vp(probs, v)
    Hv2 = HTsqrt_vp(probs, Hsqrtv)
    
    assert jnp.all(jnp.isclose(Hsqrtvd, Hsqrtv, atol=1e-6))
    assert jnp.all(jnp.isclose(Hv, Hv2, atol=1e-6))