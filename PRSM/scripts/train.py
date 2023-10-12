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

import matplotlib.pyplot as plt
from hitopt.utils.argparser import PRSMModelArgs
from hitopt.utils.pre_processing import read_dataset, in_place_shuffle, in_place_normalize, train_test_indices, \
    split_by_indices
from hitopt.utils.templates import PRSMModelConfigTemplate
from hitopt.models.prsm import PRSMModel
import hitopt.core.repo as model_manager
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

prsm_model_config = PRSMModelConfigTemplate()
prsm_model_config.data = {
    'x_train': train_feature,
    'y_train': train_label,
}
prsm_model_config.parameters = {
    'lambda': cmdargs.regularization_strength,
    'e': 0.01,
    'maxIteration': 1500,
}

repoName = 'prsm_admm'
repo_name = repoName
try:
    repo = model_manager.open_repository(repoName)
except model_manager.RepositoryNotExists:
    repo = model_manager.create_repository(repoName=repoName, identityFields=list(prsm_model_config.parameters.keys()))

prsm_model_config.configs = {
    'repo': repo,
    'verbose': True,
    'overrideUnfinished': True,
}

prsm_model = PRSMModel(model_config=prsm_model_config)

prsm_model.train()

model = repo.pick(**prsm_model_config.parameters)
print_report({
    'lambda': model['lambda'],
    'e': model['e'],
    'training_time': f"{model['training_time']:.2f}s",
    'iterations': model['iterations'],
    'final_loss': model['loss_history'][-1],
    'final_training_mse': model['training_mse_history'][-1],
    'final_nonzero_coef_num': model['nonzero_coef_num'],
    'final_testing_mse': mse(test_feature, test_label, model['coefficient']),
}, filepath=os.path.join(GlobalConfig["OUTPUS_DIR_ROOT"], f'{repo_name}_report.txt'))

fig, ax = plt.subplots(figsize=(10, 10))
ax.plot(range(1, model['iterations']), model['loss_history'][:-1],
        label=f"$lambdas={prsm_model_config.parameters['lambda']}$")

ax.set_xlabel("Iterations")
ax.set_ylabel("Loss")
ax.legend()
ax.set_yscale("log")
ax.set_title(f"Loss - Iterations")
fig.savefig(os.path.join(GlobalConfig["OUTPUS_DIR_ROOT"],
                         f"{repo_name}_loss_iterations_lambda_{prsm_model_config.parameters['lambda']}.png"))
