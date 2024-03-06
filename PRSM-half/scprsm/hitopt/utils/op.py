import jax.numpy as jnp
from jax import jit


@jit
def least_square_loss(X, Y, W):
    return 1 / 2 * jnp.sum(jnp.square(Y - X @ W))


@jit
def nuclear_norm(W, singulars=None, U=None, Vh=None):
    if singulars is None:
        U, singulars, Vh = jnp.linalg.svd(W, full_matrices=False)
    return jnp.sum(singulars), singulars, U, Vh


@jit
def singular_thresshold(W, threshold, singulars=None, U=None, Vh=None):
    if singulars is None:
        U, singulars, Vh = jnp.linalg.svd(W, full_matrices=False)
    singulars_thress = jnp.maximum(singulars - threshold, 0)
    W_thress = U * singulars_thress @ Vh
    return W_thress, singulars, U, Vh

@jit
def prox_2_1_2(W, lambda_):
    v=1/2
    def phi_fn(W, lambda_,v):
        return jnp.arccos(v*lambda_/4*jnp.power(3/jnp.linalg.norm(W, 'fro'),3/2))

    thres = 3/2*jnp.power(v*lambda_,2/3)
    fro = jnp.linalg.norm(W, 'fro')
    shared_component = 16*jnp.power(fro,3/2)*jnp.power(jnp.cos(jnp.pi/3-phi_fn(W, lambda_, v)/3),3)
    proxed_value = shared_component/(3*jnp.sqrt(3)*v*lambda_+shared_component)*W

    return jnp.where(fro >= thres, proxed_value, 0)