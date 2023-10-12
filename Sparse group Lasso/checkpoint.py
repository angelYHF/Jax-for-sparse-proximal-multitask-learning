import numpy as np
import os.path


class Checkpoint:
    def __init__(self, path, override=False):
        self.ckpFilename = path

        if not override and os.path.exists(self.ckpFilename):
            raise Exception("Checkpoint already exists.")
        self.container: dict[str:object] = {}

    def register_fields(self, fieldsList):
        for (key, obj) in fieldsList:
            self.container[key] = obj

    def __getitem__(self, item):
        if item in self.container:
            return self.container[item]
        else:
            raise Exception("Key not found.")

    def __setitem__(self, key, value):
        if key in self.container:
            self.container[key] = value
        else:
            raise Exception("Key not found.")

    def save(self):
        np.savez(self.ckpFilename, **self.container)
