import sys
import numpy as np
from ..models.prsm import PRSMModel
import hitopt.core.repo as model_manager


class PRSMKFoldEstimator:
    def __init__(self, model_config_list, num_fold, full_features, full_labels, on_model_finished_training):
        """
        Callback 'on_model_finished_training' should have the signiture of
            on_model_finished_training(self, i_model, corresponding_config, corresponding_fold, model_trained)
        """
        self.num_samples = full_features.shape[0]
        self.num_fold = num_fold
        self.model_config_list = model_config_list
        self.full_features = full_features
        self.full_labels = full_labels
        self.on_model_finished_training = on_model_finished_training
        self.repo_name = 'prsm_admm_cv'
        self.models_fold_testing_mse = np.zeros((len(model_config_list), num_fold))
        self.models_fold_training_mse = np.zeros((len(model_config_list), num_fold))
        self.models_fold_configs = [[] for _ in range(len(model_config_list))]

    def evaluate(self):
        for i_model, model_config in enumerate(self.model_config_list):
            if self.num_samples % self.num_fold != 0:
                print(
                    f"Warning: Given dataset of {self.num_samples} samples can't be equally divided into {self.num_fold} part and the remainder will be discarded.",
                    file=sys.stderr)
            interval = int(self.num_samples / self.num_fold)  # will discard last
            for i_fold in range(self.num_fold):
                train_features = self.full_features[i_fold * interval:(i_fold + 1) * interval, :]
                train_lables = self.full_labels[i_fold * interval:(i_fold + 1) * interval, :]

                model_config.parameters['i_fold'] = i_fold
                repo_name = self.repo_name
                try:
                    repo = model_manager.open_repository(repo_name)
                except model_manager.RepositoryNotExists:
                    repo = model_manager.create_repository(repoName=repo_name,
                                                           identityFields=list(model_config.parameters.keys()))
                model_config.configs['repo'] = repo
                model_config.data = {
                    'x_train': train_features,
                    'y_train': train_lables,
                }

                prsm_model = PRSMModel(model_config=model_config)
                prsm_model.train()

                self.on_model_finished_training(self, i_model, model_config, i_fold, prsm_model)
