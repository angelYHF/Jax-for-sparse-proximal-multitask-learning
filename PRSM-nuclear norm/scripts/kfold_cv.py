DATASET_DIR_BASE = "./datasets"  # Directory containing our datasets file
LIB_DIR_BASE = "./"  # Parent directory of our module
OUTPUTS_DIR_PREFIX = "./"  # Directory in which an "outputs" sub-directory will be created

import sys
sys.path.append(LIB_DIR_BASE)

import os
dataset_file_path = os.path.join(DATASET_DIR_BASE, "newdata-hitopt.npy")

from hitopt.utils.config import GlobalConfig
GlobalConfig['DATASET_DIR_BASE'] = DATASET_DIR_BASE
GlobalConfig['LIB_DIR_BASE'] = LIB_DIR_BASE
GlobalConfig['OUTPUS_DIR_ROOT'] = os.path.join(OUTPUTS_DIR_PREFIX, 'outputs')

from pprint import pprint
pprint(GlobalConfig)

import copy
import matplotlib.pyplot as plt
import numpy as np
from hitopt.utils.argparser import PRSMModelArgs
from hitopt.utils.pre_processing import read_dataset, in_place_shuffle, in_place_normalize, train_test_indices, split_by_indices
from hitopt.utils.templates import PRSMModelConfigTemplate
from hitopt.utils.estimator import PRSMKFoldEstimator
from hitopt.utils.templates import print_report
from hitopt.utils.tools import mse

cmdargs, _ = PRSMModelArgs()

npdata = read_dataset(dataset_file_path)
if cmdargs.num_subset_feature is not None:
    npdata = npdata[:, :cmdargs.num_subset_feature]
in_place_shuffle(npdata, GlobalConfig['SEED'])
labels = npdata[:, :5]  # first 5 columns
features = npdata[:, 5:]
num_of_samples = labels.shape[0]
in_place_normalize(features)
training_indices, testing_indices = train_test_indices(num_of_samples, cmdargs.testing_set_ratio)
train_feature, train_label, test_feature, test_label = split_by_indices(features, labels, training_indices,
                                                                        testing_indices)


config_template = PRSMModelConfigTemplate(verbose=False)
config_template.parameters = {
    'lambda': None,
    'e': 0.01,
    'maxIteration': 1500,
}
lambda_schedule = sorted([1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 1e4]
                         + list(np.linspace(start=180, stop=199, num=20)))
cv_config_list = []
for lambda_ in lambda_schedule:
    config_template_copy = copy.deepcopy(config_template)
    config_template_copy.parameters['lambda'] = lambda_
    cv_config_list.append(config_template_copy)


def on_model_finished_training(self, i_model, corresponding_config, i_fold, model_trained):
    model = corresponding_config.configs['repo'].pick(**corresponding_config.parameters)
    testing_mse = mse(test_feature, test_label, model['coefficient'])
    self.models_fold_testing_mse[i_model][i_fold] = testing_mse
    self.models_fold_training_mse[i_model][i_fold] = model['training_mse_history'][-1]
    self.models_fold_configs[i_model].append(corresponding_config)
    print_report({
        'i_fold': i_fold,
        'lambda': model['lambda'],
        'e': model['e'],
        'training_time': f"{model['training_time']:.2f}s",
        'iterations': model['iterations'],
        'final_loss': model['loss_history'][-1],
        'final_training_mse': model['training_mse_history'][-1],
        'final_nonzero_coef_num': model['nonzero_coef_num'],
        'final_testing_mse': testing_mse,
    }, filepath=os.path.join(GlobalConfig["OUTPUS_DIR_ROOT"], f'{self.repo_name}_report.txt'))

cv_estimator = PRSMKFoldEstimator(cv_config_list, 10, train_feature, train_label, on_model_finished_training)

cv_estimator.evaluate()

fig, ax = plt.subplots(figsize=(10, 10))
ax.plot(lambda_schedule, np.mean(cv_estimator.models_fold_testing_mse, axis=1))
for one_model_folds in cv_estimator.models_fold_testing_mse.T:
    _ = ax.scatter(lambda_schedule, one_model_folds, s=10, c='gray')

ax.scatter(lambda_schedule, np.mean(cv_estimator.models_fold_testing_mse, axis=1), s=20, c='r')

ax.set_xlabel("$\lambda$")
ax.set_ylabel("Average Testing MSE")
ax.set_xscale("log")
ax.set_yscale("log")
ax.grid()
ax.set_title(f"10-Fold Cross Validation")
fig.savefig(os.path.join(GlobalConfig["OUTPUS_DIR_ROOT"],
                         f"{cv_estimator.repo_name}_cross_validation_testing_mse_{lambda_schedule[0]}-{lambda_schedule[-1]}_{len(lambda_schedule)}_lambdas.png"))

fig, ax = plt.subplots(figsize=(10, 10))
ax.plot(lambda_schedule, np.mean(cv_estimator.models_fold_training_mse, axis=1))
ax.scatter(lambda_schedule, np.mean(cv_estimator.models_fold_training_mse, axis=1), s=10, c='r')

ax.set_xlabel("$\lambda$")
ax.set_ylabel("Average Training MSE")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_title(f"10-Fold Cross Validation")
fig.savefig(os.path.join(GlobalConfig["OUTPUS_DIR_ROOT"],
                         f"{cv_estimator.repo_name}_cross_validation_training_mse_{lambda_schedule[0]}-{lambda_schedule[-1]}_{len(lambda_schedule)}_lambdas.png"))


cv_estimator.model_loss_len_grid = np.zeros((len(cv_estimator.models_fold_configs), len(cv_estimator.models_fold_configs[0])))

for i_model, config_row in enumerate(cv_estimator.models_fold_configs):
    for i_fold, model_config in enumerate(config_row):
        model = model_config.configs['repo'].pick(**model_config.parameters)
        cv_estimator.model_loss_len_grid[i_model][i_fold] = len(model["loss_history"])

cv_estimator.model_loss_len_fold_min = np.min(cv_estimator.model_loss_len_grid, axis=1)
cv_estimator.model_loss_fold_averaged = []

for i_model, config_row in enumerate(cv_estimator.models_fold_configs):
    cv_estimator.model_loss_fold_averaged.append(np.zeros(int(cv_estimator.model_loss_len_fold_min[i_model])))
    fold_count = 0
    for i_fold, model_config in enumerate(config_row):
        model = model_config.configs['repo'].pick(**model_config.parameters)
        cv_estimator.model_loss_fold_averaged[i_model] += model["loss_history"][:int(cv_estimator.model_loss_len_fold_min[i_model])]
        fold_count+=1
    cv_estimator.model_loss_fold_averaged[i_model]/=fold_count

fig, ax = plt.subplots(figsize=(12, 10))

for i_model, config_row in enumerate(cv_estimator.models_fold_configs):
    model_config = config_row[0]
    model = model_config.configs['repo'].pick(**model_config.parameters)
    _ = ax.plot(range(1, len(cv_estimator.model_loss_fold_averaged[i_model])), cv_estimator.model_loss_fold_averaged[i_model][:-1],
    label=f"$lambdas={model_config.parameters['lambda']}$")


ax.set_xlabel("Iterations")
ax.set_ylabel("Loss")
ax.set_xscale("log")
ax.set_yscale("log")
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
ax.set_title(f"10-Fold Cross Validation Loss Comparision")
fig.tight_layout()
fig.savefig(os.path.join(GlobalConfig["OUTPUS_DIR_ROOT"],
                         f"{cv_estimator.repo_name}_loss_iterations_{lambda_schedule[0]}-{lambda_schedule[-1]}_{len(lambda_schedule)}_lambdas.png"))
