import numpy as np
import codecs
import sys


class ModelConfigTemplate:
    pass


class PRSMModelConfigTemplate(ModelConfigTemplate):
    def __init__(self, verbose=True):
        self.data = {
            'x_train': np.arange(0),
            'y_train': np.arange(0),
        }
        self.parameters = {  # parameters that distinguish two models
            'lambda': 0,
            'e': 0,
            'maxIteration': 0,
        }
        self.configs = {
            'repo': None,
            'verbose': True,
            'overrideUnfinished': True,
        }
        if verbose:
            self.report(template=True)

    def report(self, template=False):
        print('-' * 10)
        print(f'Model Config{" Template" if template else ""}:')
        print(f'Data:')
        for k, v in self.data.items():
            print(f'\t', end='')
            print(f'{k}: shape {v.shape}')
        print(f'Parameters:')
        for k, v in self.parameters.items():
            print(f'\t', end='')
            print(f'{k}: {v}')
        print(f'Configs:')
        for k, v in self.configs.items():
            print(f'\t', end='')
            print(f'{k}: {v}')
        print('-' * 10)


def print_report_to_device(content_dict, device):
    print('-' * 10, file=device)
    for k, v in content_dict.items():
        print(f'{k}={v}', file=device)
    print('-' * 10, file=device)


def print_report(content_dict, filepath=None):
    print_report_to_device(content_dict, sys.stdout)
    if filepath is not None:
        with codecs.open(filepath, 'a', 'utf8') as f:
            print_report_to_device(content_dict, f)
        print(f"Log file saved to {filepath}")
