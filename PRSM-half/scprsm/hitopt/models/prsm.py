import numpy as np
import time
from tqdm.auto import tqdm
from functools import partial
import jax.numpy as jnp
from jax import grad, jit
from jax import device_put
from ..core.basemodel import BaseModel
from ..utils.tools import mse
from ..utils.op import least_square_loss, prox_2_1_2


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

        self.rho = 1
        self.gamma = 1
        self.alpha = 1.2
        self.num_samples = self._x_train_shape[0]

        # self.Cache = {
        #     "XTY": self.X.T @ self.Y,
        #     "INVXTXbetaI": jnp.linalg.inv(
        #         self.X.T @ self.X +
        #         self.rho * jnp.eye(self._x_train_shape[1])
        #     ),
        # }

    @partial(jit, static_argnums=(0,))
    def _loss(self, w):
        L2_1_2 = jnp.square(
            jnp.sum(jnp.power(jnp.sum(jnp.square(w), axis=1), 1/4)))
        return least_square_loss(self.X, self.Y, w) + self.lambda_ * L2_1_2

    @partial(jit, static_argnums=(0,))
    def _prsm_step(self, B, L):
        # Z = self.Cache["INVXTXbetaI"] @ (self.Cache["XTY"] +
                                        #  self.gamma * B + L)
        Z = prox_2_1_2(B+L, self.lambda_)
        L = L-self.alpha*self.gamma*(Z-B)
        B = prox_2_1_2(Z-1/self.gamma*L, self.lambda_)
        L = L-self.alpha*self.gamma*(Z-B)
        return B, L

    def _train_(self):
        """
        This method doesn't have a returned value.
        An early returning may happen in __init_training_() or _close_training_().
        """
        L = jnp.zeros_like(self.w)
        B = self.w

        iters = 0
        t0 = time.time()
        pbar = tqdm(total=self.maxIteration)
        pbar.set_description("Training...")
        while iters < self.maxIteration:
            B, L = self._prsm_step(B, L)

            stopping_criterion = jnp.linalg.norm(B - L, 'fro')
            if stopping_criterion < self.e:
                break

            # if jnp.count_nonzero(self.w)<1000:
            #     break

            self.w = B
            loss = self._loss(self.w)
            training_mse = mse(self.X, self.Y, self.w)

            self.checkpoint['loss_history'].append(loss)
            self.checkpoint['training_mse_history'].append(training_mse)

            iters += 1
            pbar.set_postfix(loss=loss, mse=training_mse,
                             stopping=stopping_criterion)
            pbar.update()
        pbar.close()

        self.checkpoint['iterations'] = iters
        self.checkpoint['training_time'] = time.time() - t0
        self.checkpoint['coefficient'] = self.w
        self.checkpoint['nonzero_coef_num'] = jnp.count_nonzero(self.w)
