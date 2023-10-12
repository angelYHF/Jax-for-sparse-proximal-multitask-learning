import os.path
import numpy as np
import psutil
import jax.numpy as jnp
from jax import device_put


def calculate_lipschitz(X):
    X = device_put(X)
    q = jnp.dot(X.T, X)
    return jnp.linalg.norm(q, 'fro')


def get_memory_usage_GiB():
    return psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3


def normalize(numpy_ndarray):
    mu = np.mean(numpy_ndarray)
    std = np.std(numpy_ndarray)
    return (numpy_ndarray - mu) / std


def split_x_y(*args):
    res = []
    for dataSets in args:
        res.append(dataSets[:, 2:])
        res.append(dataSets[:, :2])
    return res


def __uniform(shape):
    return np.random.uniform(low=-0.3, high=0.3, size=shape)


def __normal(shape):
    return np.random.normal(loc=2, scale=1, size=shape)


__initializationMethodMapper = {
    "uniform": __uniform,
    "normal": __normal,
}


def initialize_random_weights(shape: tuple, dist="uniform"):
    filename = f"initial_weights_{dist}_{shape}.npz"
    if os.path.exists(filename):
        return np.load(filename)["arr_0"]
    else:
        w = __initializationMethodMapper[dist](shape)
        np.savez(filename, w)
    return w
