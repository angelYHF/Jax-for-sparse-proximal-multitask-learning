import os
import numpy as np


class _FilenameMapper:
    def __init__(self, hyperParamMark: str):
        self.hyperParamMark = hyperParamMark
        self.paramDict = {"lr_lambda": ['lr', 'lambda_'],
                          "lambda": ['lambda_'],
                          "e_lambda": ['e', 'lambda_'],
                          "cv_lambda": ['fold', 'lambda_'],
                          "best_lambda": ['bestLambda']}
        self.paramRequired = self.paramDict[hyperParamMark]

    def get_filename(self, kwargs):
        self._check_param(kwargs)
        keywordList = list(kwargs.keys())
        keywordList.sort()
        return self._encode_filename(keywordList, [kwargs[k] for k in keywordList])

    def _check_param(self, kwargs):
        if len(kwargs) != len(self.paramRequired):
            raise Exception("Parameter numbers mismatch to definition!")
        for paramStr in self.paramRequired:
            if not kwargs.__contains__(paramStr):
                raise Exception(f"Parameter '{paramStr}' is unspecified!")

    def _encode_filename(self, keywordList, valueList):
        filename = ""
        for key, val in zip(keywordList, valueList):
            filename += f"{key}_{float(val)}_"
        return filename


class CheckpointManager:
    def __init__(self, hyperParamMark):
        self.filenameMapper = _FilenameMapper(hyperParamMark)
        self._prefix = f"checkpoint/{hyperParamMark}/"

    def _save_checkpoint(self, name, dataDict):
        filename = self._prefix + name + ".npz"
        try:
            np.savez(filename, **dataDict)
        except FileNotFoundError:
            os.makedirs(os.path.dirname(filename))
            np.savez(filename, **dataDict)

    def _load_checkpoint(self, name):
        filename = self._prefix + name + ".npz"
        try:
            return np.load(filename)
        except FileNotFoundError:
            if not os.path.exists(os.path.dirname(filename)):
                os.makedirs(os.path.dirname(filename))
            return np.load(filename)

    def load_checkpoint(self, **kwargs):
        filename = self.filenameMapper.get_filename(kwargs)
        return self._load_checkpoint(filename)

    def save_checkpoint(self, dataDict, **kwargs):
        filename = self.filenameMapper.get_filename(kwargs)
        self._save_checkpoint(filename, dataDict)

    def query_checkpoint_id(self, **kwargs):
        name = self.filenameMapper.get_filename(kwargs)
        return self._prefix + name + ".npz"
