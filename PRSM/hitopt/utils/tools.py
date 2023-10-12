import numpy as np
import os


def schedule_parameter(listdata, precision=5):
    return np.array([round(x, precision) for x in listdata])


def mse(X, Y, w):
    Y_hat = X @ w
    return np.mean(np.square(Y_hat - Y))


def npload(filename):
    return dict(np.load(filename))


def identity_to_filename(identityDict):
    filename = ""
    keysList = list(identityDict.keys())
    keysList.sort()

    for key in keysList:
        filename += f"{key}_{float(identityDict[key])}_"
    return filename + '.npz'


def normalize(numpy_ndarray, axis):
    mu = np.mean(numpy_ndarray, axis=axis, keepdims=True)
    std = np.std(numpy_ndarray, axis=axis, keepdims=True)
    return (numpy_ndarray - mu) / std


def split_x_y(*args):
    res = []
    for dataSets in args:
        res.append(dataSets[:, 2:])
        res.append(dataSets[:, :2])
    return res


def calculate_lipschitz(X):
    q = np.dot(X.T, X)
    return np.linalg.norm(q, 'fro')


def protect(*protected):
    """
    Returns a metaclass that protects all attributes given as strings
    """

    class Protect(type):
        has_base = False

        def __new__(meta, name, bases, attrs):
            if meta.has_base:
                for attribute in attrs:
                    if attribute in protected:
                        raise AttributeError('Overriding of attribute "%s" not allowed.' % attribute)
            meta.has_base = True
            klass = super().__new__(meta, name, bases, attrs)
            return klass

    return Protect


def smooth(series, weight):
    """
        weight: the bigger the weight is, the smoother the output will be
    """
    assert len(series) >= 3, f"The to be smoothed serires contains only {len(series)} data points"
    smoothed = [(1 - weight) * series[0]]
    for i in range(1, len(series)):
        smoothed.append(weight * smoothed[i - 1] + (1 - weight) * series[i])

    return smoothed


def switch_model(name):
    modelFilename = 'model.py'
    if os.path.exists(modelFilename):
        os.remove(modelFilename)
    os.symlink(f'dev/{name}.py', 'model.py')
