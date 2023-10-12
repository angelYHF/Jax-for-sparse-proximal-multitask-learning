import os
import random
import time
import numpy
from tqdm import tqdm
from functools import partial
import jax.numpy as jnp
from jax import grad, jit
from jax import device_put
import tools


class TrainingCompletedException(Exception):
    def __init__(self, message):
        self.message = message


class MemoryExceedsException(Exception):
    def __init__(self, message):
        self.message = message


class RegressionModel:
    def __init__(self, x_train, y_train, initCoef, lambda_, checkpointCfg, maxIteration=4000, tolerance=1e-8,
                 verbose=True):
        self.checkpointCfg = checkpointCfg
        self._mgr, self._mgrParams = checkpointCfg["checkpointMgrRef"], checkpointCfg["checkpointParams"]
        if os.path.exists(self._mgr.query_checkpoint_id(**self._mgrParams)):
            raise TrainingCompletedException("The model had been trained and results snapshot detected!")
        else:
            if verbose:
                print(f"\nmodel({self._mgr.query_checkpoint_id(**self._mgrParams)}) starts training.\n")

        self.w = device_put(initCoef)
        self.lambda_ = lambda_
        self.maxIteration = maxIteration
        self.tolerance = device_put(tolerance)
        self.verbose = verbose

        self._x_train_shape = x_train.shape
        self._y_train_shape = y_train.shape
        self.X = device_put(x_train)
        self.Y = device_put(y_train)
        if verbose:
            print(f"Float Precision: {self.Y.dtype}")

        self.checkpoint = {
            "lambda": lambda_,
            "tolerance": tolerance,
            "coefficient": None,
            "nonzero_coef_num": -1,
            "iterations": -1,
            "training_time": -1.,
            "testing_mse": None,

            "loss_history": [],
            "training_mse_history": [],
            "learning_rate_history": [],
        }

    def train(self, print_every):
        if tools.get_memory_usage_GiB() > 10:
            raise MemoryExceedsException(f"{tools.get_memory_usage_GiB():.2f} GiB memory has been used!")
        t0 = time.time()
        self.checkpoint['iterations'] = self._train(print_every)
        training_time = time.time() - t0
        self.checkpoint['training_time'] = training_time
        self.checkpoint['coefficient'] = self.w
        self._regularize_result()
        if self.verbose:
            self._report_result()

    def test(self, x_test, y_test):
        mse = self._mse(x_test, y_test, self.w)
        self.checkpoint["testing_mse"] = mse
        return mse

    def save_checkpoint(self):
        self._mgr.save_checkpoint(self.checkpoint, **self._mgrParams)

    def _train(self, print_every):
        raise NotImplementedError

    def _loss(self, w):
        raise NotImplementedError

    def _mse(self, X, Y, w):
        raise NotImplementedError

    def _regularize_result(self):
        iters = self.checkpoint["iterations"]
        self.checkpoint['coefficient'] = numpy.array(self.checkpoint['coefficient'])
        self.checkpoint['loss_history'] = numpy.array(self.checkpoint['loss_history'][-iters:])
        self.checkpoint['learning_rate_history'] = numpy.array(self.checkpoint['learning_rate_history'][-iters:])
        self.checkpoint['training_mse_history'] = numpy.array(self.checkpoint['training_mse_history'][-iters:])
        self.checkpoint['nonzero_coef_num'] = numpy.array(self.checkpoint['nonzero_coef_num'])

    def _report_result(self):
        print(f"----------\n"
              f"\tlambda={self.checkpoint['lambda']}\n"
              f"\te={self.checkpoint['tolerance']}\n"
              f"\ttraining_time={self.checkpoint['training_time']:.2f} sec\n"
              f"\tepochs={self.checkpoint['iterations']}\n"
              f"\tfinal_loss={self.checkpoint['loss_history'][-1]}\n"
              f"\tfinal_training_mse={self.checkpoint['training_mse_history'][-1]}\n"
              f"\tfinal_learning_rate={self.checkpoint['learning_rate_history'][-1]}\n"
              f"\tfinal_nonzero_coef_num={self.checkpoint['nonzero_coef_num']}\n"
              f"----------\n\n"
              )


class MultiTaskGroupLassoISTA(RegressionModel):
    def __init__(self, x_train, y_train, initCoef, lambda_, checkpointCfg, maxIteration=4000, tolerance=1e-8,
                 verbose=True):
        super().__init__(x_train, y_train, initCoef, lambda_, checkpointCfg, maxIteration, tolerance, verbose)
        self.checkpoint['loss_history'].append(self._loss(self.w))
        self.checkpoint['training_mse_history'].append(self._mse(self.X, self.Y, self.w))
        self._lastLearningRate = 1

    @partial(jit, static_argnums=(0,))
    def _loss(self, w):
        return self._loss_g(w) + self.lambda_ * jnp.sum(jnp.linalg.norm(w, axis=1))

    @partial(jit, static_argnums=(0,))
    def _loss_g(self, w):
        return 1 / 2 * jnp.sum(jnp.square(self.Y - self.X @ w))

    @partial(jit, static_argnums=(0,))
    def _gradient_loss_g(self, w):
        return self.X.T @ (self.X @ w - self.Y)

    @partial(jit, static_argnums=(0,))
    def _groupsoft(self, lr, matrix):
        normed_vec = jnp.linalg.norm(matrix, axis=1, keepdims=True)
        return jnp.maximum(normed_vec - lr * self.lambda_, 0) / normed_vec * matrix

    @partial(jit, static_argnums=(0,))
    def _mse(self, X, Y, w):
        Y_hat = X @ w
        mse = jnp.mean(jnp.reshape(jnp.square(Y_hat - Y), (-1, self._y_train_shape[1])), axis=0)
        return mse

    def _backtracking_line_search(self, w):
        a = 0.8
        lr = self._lastLearningRate / random.uniform(a / 2, a)

        gW = self._loss_g(w)
        d_gW = self._gradient_loss_g(w)
        prox_w = self._groupsoft(lr, w - lr * d_gW)
        delta = w - prox_w

        while self._loss_g(prox_w) - gW > jnp.sum(-d_gW * delta) + 1 / lr / 2 * jnp.sum(jnp.square(delta)):
            lr = a * lr
            prox_w = self._groupsoft(lr, w - lr * d_gW)
            delta = w - prox_w

        self._lastLearningRate = lr
        return prox_w, lr

    def _train(self, print_every):
        iters = 0
        t0 = time.time()
        with tqdm(total=self.maxIteration) as pbar:
            pbar.set_description(f'e={self.tolerance}, Lambda={self.lambda_}')
            while iters < self.maxIteration:
                prox_w, lr = self._backtracking_line_search(self.w)
                #改进的停止条件
                if jnp.linalg.norm(self.w - prox_w, 'fro') / jnp.linalg.norm(self.w, 'fro') < self.tolerance \
                        or jnp.sum(self.w) == 0:
                    break

                self.w = prox_w
                loss = self._loss(self.w)
                self.checkpoint['loss_history'].append(loss)
                self.checkpoint['training_mse_history'].append(self._mse(self.X, self.Y, self.w))
                self.checkpoint['learning_rate_history'].append(lr)
                iters += 1

                if print_every and 0 == iters % print_every:
                    print(
                        f"iters={iters},loss={loss},lr={lr},"
                        f" Elapsed {time.time() - t0 :.2f} sec")
                    pbar.update(print_every)

        self.checkpoint['nonzero_coef_num'] = jnp.count_nonzero(self.w)
        return iters
