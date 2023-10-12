import numpy as np
import os.path


class DataLoader:
    def __init__(self, dataFilename):
        self.data = np.loadtxt(dataFilename, dtype=float, delimiter=',')
        self.data = self.shuffle(self.data)

    @staticmethod
    def shuffle(data):
        bak=data
        np.random.default_rng().shuffle(bak)
        return bak

    def split_cv_train_val_test(self, testRatio, cvFold, regenerate=False):
        dataFilename = f"{cvFold}_fold_cv_data_split.npz"
        if regenerate:
            self.data = self.shuffle(self.data)
            if os.path.exists(dataFilename):
                os.remove(dataFilename)

        if os.path.exists(dataFilename):
            f = np.load(dataFilename)
            trainValDataDict = {}
            for i in range(f["keysNum"]):
                trainValDataDict[i] = (f["train" + str(i)], f["val" + str(i)])
            test = f["test"]
        else:
            train, test = self.split_train_test(testRatio=testRatio, fixed=False, trainingRoundBy=cvFold)
            trainValDataDict = self._split_train_val_crossed(cvFold, train)
            dictKeysNum = len(trainValDataDict.keys())
            np.savez(dataFilename, keysNum=dictKeysNum,
                     **{"train" + str(k): v[0] for k, v in trainValDataDict.items()},
                     **{"val" + str(k): v[1] for k, v in trainValDataDict.items()},
                     test=test)
        return trainValDataDict, test

    def split_train_test(self, testRatio: float, fixed: bool, trainingRoundBy=1):
        if fixed:
            trainingRatio = 1 - testRatio
            filename = f"train{trainingRatio}_test{testRatio}.npz"
            if os.path.exists(filename):
                m = np.load(filename)
                train, test = m["train"], m["test"]
                print(f"read split result from {filename} ...")
            else:
                train, test = self._split_train_test_random(testRatio, trainingRoundBy)
                np.savez(filename, **{"train": train, "test": test})
            return train, test
        else:
            return self._split_train_test_random(testRatio, trainingRoundBy)

    def _split_train_test_random(self, testRatio, trainingRoundBy=1):
        trainingRatio = 1 - testRatio
        assert 0. < trainingRatio < 1.

        totalSize = len(self.data)
        trainingSize = int(totalSize * trainingRatio)
        trainingSize = trainingSize // trainingRoundBy * trainingRoundBy
        testSize = totalSize - trainingSize

        train, test = self.data[:trainingSize], \
                      self.data[-testSize:]

        return train, test

    @staticmethod
    def _split_train_val_crossed(cvFold, data):
        resDict = {}
        if len(data) % cvFold != 0:
            raise Exception("num_data_points is not divisible by fold_k")
        splitList = np.split(data, cvFold)
        for i, val in enumerate(splitList):
            trainSet = np.concatenate((splitList[:i] + splitList[i + 1:]))
            valSet = splitList[i]
            resDict[i] = (trainSet, valSet)
        return resDict
