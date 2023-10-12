import numpy as np


def load_csv_data(csvFilePath):
    """
    Load dataset from csv file.
    Return a Dataset instance.
    """
    dataset = Dataset()
    dataset.data = np.loadtxt(csvFilePath, dtype=float, delimiter=',')

    return dataset


class Dataset:
    def __init__(self):
        self.data = None

    def shuffle(self):
        """
        Randomly shuffle the data.
        """
        np.random.default_rng().shuffle(self.data)

    def split(self, testRatio, valSet=False, valRatio=0, trainingRoundBy=1):
        """
        Perform fixed split only.
        """
        if not valSet:
            trainingRatio = 1 - testRatio
            assert 0. < trainingRatio < 1.

            totalSize = len(self.data)
            trainingSize = int(totalSize * trainingRatio)
            trainingSize = trainingSize // trainingRoundBy * trainingRoundBy
            testSize = totalSize - trainingSize

            train, test = self.data[:trainingSize], \
                          self.data[-testSize:]

            return train, test
        else:
            raise NotImplementedError

    def cv_split(self, cvFold, testRatio, valSet=False, valRatio=0):
        raise NotImplementedError
