import jax.numpy as jnp
from jax import jit


@jit
def least_square_loss(X, Y, W):
    n_samples = X.shape[0]
    return 1 / n_samples * jnp.sum(jnp.square(Y - X @ W))


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
