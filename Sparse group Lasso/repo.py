import os
import yaml
import codecs
from tools import identity_to_filename, npload


class RepositoryNotExists(Exception):
    def __init__(self, message):
        self.message = message


class RepositoryExists(Exception):
    def __init__(self, message):
        self.message = message


repositoryBaseDir = os.path.join(os.path.dirname(__file__), "checkpoints")


class Repository:
    def __init__(self, repoName, identityFields):
        self.repoName = repoName
        self.identityFields = identityFields
        self.repoPath = os.path.join(repositoryBaseDir, repoName)

    def pick(self, pathOnly=False, **kwargs):
        if len(kwargs) != len(self.identityFields):
            raise Exception("Parameter numbers mismatch to definition!")
        for paramStr in self.identityFields:
            if not kwargs.__contains__(paramStr):
                raise Exception(f"Parameter '{paramStr}' is unspecified!")

        filepath = os.path.join(self.repoPath, identity_to_filename(kwargs))
        if pathOnly:
            return filepath
        else:
            return npload(filepath)

    def models(self):
        """
        return an iterable of this repo
        """
        self.checkpointFilenames = os.listdir(self.repoPath)
        self.filenameIterator = iter(self.checkpointFilenames)
        return self

    def __iter__(self):
        return self

    def __next__(self):
        filename = next(self.filenameIterator)
        while not filename.endswith('.npz'):
            filename = next(self.filenameIterator)
        file = npload(os.path.join(self.repoPath, filename))
        while not file['finished']:
            filename = next(self.filenameIterator)
            file = npload(os.path.join(self.repoPath, filename))
        return file


def open_repository(repoName):
    repoPath = os.path.join(repositoryBaseDir, repoName)
    if not os.path.exists(repoPath):
        raise RepositoryNotExists(f"Repo {repoName} doesn't exit.")
    with codecs.open(os.path.join(repoPath, 'repo.yaml'), 'r', 'utf8') as f:
        yamlContext = yaml.load_all(f, Loader=yaml.SafeLoader)
        yamlDict = next(yamlContext)

    return Repository(yamlDict['repoName'], yamlDict['identityFields'])


def create_repository(repoName, identityFields):
    repoPath = os.path.join(repositoryBaseDir, repoName)
    if os.path.exists(repoPath):
        raise RepositoryExists(f"Repo {repoName} already exists.")
    else:
        os.makedirs(repoPath)
    yamlDict = {
        'repoName': repoName,
        'identityFields': identityFields,
    }
    with codecs.open(os.path.join(repoPath, 'repo.yaml'), 'w', 'utf8') as f:
        yaml.dump(yamlDict, f)

    return Repository(repoName, identityFields)
