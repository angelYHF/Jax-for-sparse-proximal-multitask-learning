import numpy as np
import random
import time
from tqdm import tqdm
from functools import partial
import jax.numpy as jnp
from jax import grad, jit
from jax import device_put
from basemodel import BaseModel
from tools import mse


class MixedLassoISTA(BaseModel):
    def _initialize_(self, *args, **kwargs):
        data = kwargs['data']
        params = kwargs['params']
        configs = kwargs['configs']

        self._x_train_shape = data['x_train'].shape
        self._y_train_shape = data['y_train'].shape

        self.X = device_put(data['x_train'])
        self.Y = device_put(data['y_train'])

        self.w = device_put(
            np.random.uniform(low=-0.3, high=0.3, size=(self._x_train_shape[1], self._y_train_shape[1])))

        self.lambda1 = params['lambda1']
        self.lambda2 = params['lambda2']
        self.e = device_put(params['e'])
        self.maxIteration = configs['maxIteration']
        self.verbose = configs['verbose']

        if self.verbose:
            print(f"Float Precision: {self.Y.dtype}")

        self.checkpoint.register_fields([
            ("coefficient", -1),
            ("nonzero_coef_num", -1),
            ("iterations", -1),
            ("training_time", -1.),
            ("loss_history", []),
            ("training_mse_history", []),
            ("learning_rate_history", []),
        ])

        self._linesearchLastLearningRate = 1

    @partial(jit, static_argnums=(0,))
    def _loss(self, w):
        return self._loss_g(w) + self.lambda1 * jnp.sum(jnp.linalg.norm(w, axis=1)) + self.lambda2 * jnp.sum(jnp.abs(w))

    @partial(jit, static_argnums=(0,))
    def _loss_g(self, w):
        return 1 / 2 * jnp.sum(jnp.square(self.Y - self.X @ w))

    @partial(jit, static_argnums=(0,))
    def _gradient_loss_g(self, w):
        return self.X.T @ (self.X @ w - self.Y)

    @partial(jit, static_argnums=(0,))
    def _groupsoft(self, lr, matrix):
        normed_vec = jnp.linalg.norm(matrix, axis=1, keepdims=True)
        return jnp.maximum(normed_vec - lr * self.lambda1, 0) / normed_vec * matrix

    @partial(jit, static_argnums=(0,))
    def _lassosoft(self, lr, matrix):
        return jnp.sign(matrix) * jnp.maximum(0, jnp.abs(matrix) - lr * self.lambda2)

    def _backtracking_line_search(self, w):
        a = 0.8
        lr = self._linesearchLastLearningRate / random.uniform(a / 2, a)

        gW = self._loss_g(w)
        d_gW = self._gradient_loss_g(w)
        prox_w = self._groupsoft(lr, w - lr * d_gW)
        delta = w - prox_w

        while self._loss_g(prox_w) - gW > jnp.sum(-d_gW * delta) + 1 / lr / 2 * jnp.sum(jnp.square(delta)):
            lr = a * lr
            prox_w = self._groupsoft(lr, w - lr * d_gW)
            delta = w - prox_w
        # Group Lasso step和Multi Lasso step使用相同的学习率
        d_gW = self._gradient_loss_g(prox_w)
        prox_w = self._lassosoft(lr, prox_w - lr * d_gW)

        self._linesearchLastLearningRate = lr
        return prox_w, lr

    def _train_(self, print_every):
        """
        This method doesn't have a returned value.
        An early returning may happen in __init_training_() or _close_training_().
        """

        iters = 0
        t0 = time.time()

        with tqdm(total=self.maxIteration) as pbar:
            pbar.set_description(f'e={self.e}, Lambdas={(self.lambda1, self.lambda2)}')
            while iters < self.maxIteration:
                prox_w, lr = self._backtracking_line_search(self.w)

                if jnp.linalg.norm(self.w - prox_w, 'fro') / jnp.linalg.norm(self.w, 'fro') < self.e \
                        or jnp.sum(self.w) == 0:
                    break

                self.w = prox_w
                loss = self._loss(self.w)
                self.checkpoint['loss_history'].append(loss)
                self.checkpoint['training_mse_history'].append(mse(self.X, self.Y, self.w))
                self.checkpoint['learning_rate_history'].append(lr)
                iters += 1

                # control progress bar displaying
                if iters == 10 and print_every > 10:
                    pbar.update(10)
                elif iters==print_every:
                    pbar.update(print_every-10)
                else:
                    if print_every and 0 == iters % print_every:
                        pbar.update(print_every)

        self.checkpoint['iterations'] = iters
        self.checkpoint['training_time'] = time.time() - t0
        self.checkpoint['coefficient'] = self.w
        self.checkpoint['nonzero_coef_num'] = jnp.count_nonzero(self.w)

        if self.verbose:
            print(f"----------\n"
                  f"\t(lambda1={self.checkpoint['lambda1']}, lambda2={self.checkpoint['lambda2']}, e={self.checkpoint['e']})\n"
                  f"\ttraining_time={self.checkpoint['training_time']:.2f} sec\n"
                  f"\tepochs={self.checkpoint['iterations']}\n"
                  f"\tfinal_loss={self.checkpoint['loss_history'][-1]}\n"
                  f"\tfinal_training_mse={self.checkpoint['training_mse_history'][-1]}\n"
                  f"\tfinal_learning_rate={self.checkpoint['learning_rate_history'][-1]}\n"
                  f"\tfinal_nonzero_coef_num={self.checkpoint['nonzero_coef_num']}\n"
                  f"----------\n\n"
                  )
