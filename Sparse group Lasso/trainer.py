import os
import sys
import psutil
from basemodel import BaseModel


class MultiProcessModelTrainer:
    """
    Model proxy class for memory usage control.
    """

    def __init__(self, realModelInstance: BaseModel):
        self.realModelInstance = realModelInstance

    def train(self, *args, **kwargs):
        pid = os.fork()
        # Child process
        if pid == 0:
            # 使用双精度
            os.environ["JAX_ENABLE_X64"] = 'True'
            from jax.config import config
            config.update("jax_enable_x64", True)

            self.realModelInstance.train(*args, **kwargs)
            sys.exit(0)

        # Parent process
        if pid != 0:
            # make sure that nothing gets out of control
            children = psutil.Process(os.getpid()).children()
            assert len(children) == 1

            def on_terminate(proc):
                if proc.returncode != 0:
                    raise Exception("Abnormal child process termination...")

            psutil.wait_procs(children, callback=on_terminate)
