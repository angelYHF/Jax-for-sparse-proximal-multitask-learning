import numpy as np
import time
from tqdm.auto import tqdm
from functools import partial
import jax.numpy as jnp
from jax import grad, jit
from jax import device_put
from ..core.basemodel import BaseModel
from ..utils.tools import mse
from ..utils.op import least_square_loss, nuclear_norm, singular_thresshold


class PRSMModel(BaseModel):
    def _initialize_(self, *args, **kwargs):
        kwargs['model_config'].report()
        data = kwargs['model_config'].data
        params = kwargs['model_config'].parameters
        configs = kwargs['model_config'].configs

        self.verbose = configs['verbose']
        self.maxIteration = params['maxIteration']
        print("Model Initializing...")

        self._x_train_shape = data['x_train'].shape
        self._y_train_shape = data['y_train'].shape

        self.X = device_put(data['x_train'])
        self.Y = device_put(data['y_train'])

        if self.verbose:
            print(f"Float Precision: {self.Y.dtype}")

        self.w = device_put(
            np.random.uniform(low=-3, high=3, size=(self._x_train_shape[1], self._y_train_shape[1])))

        self.lambda_ = params['lambda']
        self.e = device_put(params['e'])

        self.checkpoint.register_fields([
            ("coefficient", -1),
            ("nonzero_coef_num", -1),
            ("iterations", -1),
            ("training_time", -1.),
            ("loss_history", []),
            ("training_mse_history", []),
            ("learning_rate_history", []),
        ])

        self.beta = 1
        self.alpha = 0.9
        self.num_samples = self._x_train_shape[0]

        self.Cache = {
            "2/nXTY": 2 / self.num_samples * self.X.T @ self.Y,
            "INVXTXbetaI": jnp.linalg.inv(
                2 / self.num_samples * self.X.T @ self.X +
                self.beta * jnp.eye(self._x_train_shape[1])
            ),
        }

    @partial(jit, static_argnums=(0,))
    def _loss(self, w, singulars=None, U=None, Vh=None):
        n_norm, singulars, U, Vh = nuclear_norm(w, singulars, U, Vh)
        return least_square_loss(self.X, self.Y, w) + n_norm

    @partial(jit, static_argnums=(0,))
    def _prsm_step(self, thetay, rho):
        beta = self.beta
        alpha = self.alpha

        thetax = self.Cache["INVXTXbetaI"] @ (beta * thetay + rho + self.Cache["2/nXTY"])
        rho = rho - alpha * beta * (thetax - thetay)
        thetay, singulars, U, Vh = singular_thresshold(thetax - rho / beta, self.lambda_ / beta)
        rho = rho - alpha * beta * (thetax - thetay)

        return thetax, thetay, rho, singulars, U, Vh

    def _train_(self):
        """
        This method doesn't have a returned value.
        An early returning may happen in __init_training_() or _close_training_().
        """
        thetay = self.w
        rho = device_put(0)

        iters = 0
        t0 = time.time()
        pbar = tqdm(total=self.maxIteration)
        pbar.set_description("Training...")
        while iters < self.maxIteration:
            thetax, thetay, rho, singulars, U, Vh = self._prsm_step(thetay, rho)

            stopping_criterion = jnp.linalg.norm(thetax - thetay, 'fro')
            if stopping_criterion < self.e:
                break

            if iters > 2 and self.checkpoint['loss_history'][-1] - self.checkpoint['loss_history'][-2] > 0:
                break

            self.w = thetay
            loss = self._loss(self.w, singulars, U, Vh)
            training_mse = mse(self.X, self.Y, self.w)

            self.checkpoint['loss_history'].append(loss)
            self.checkpoint['training_mse_history'].append(training_mse)

            iters += 1
            pbar.set_postfix(loss=loss, mse=training_mse, stopping=stopping_criterion)
            pbar.update()
        pbar.close()

        self.checkpoint['iterations'] = iters
        self.checkpoint['training_time'] = time.time() - t0
        self.checkpoint['coefficient'] = self.w
        self.checkpoint['nonzero_coef_num'] = jnp.count_nonzero(self.w)
