import os
import random
import sys
import psutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lib.ista import MultiLassoIstaLineSearch, ModelExistsException, MemoryExceedsException


def normalize(numpy_ndarray):
    mu = np.mean(numpy_ndarray)
    std = np.std(numpy_ndarray)
    return (numpy_ndarray - mu) / std


def train_test_split(dataSet, testRatio, roundBy):
    np.random.default_rng().shuffle(dataSet, axis=0)
    totalSize = len(dataSet)
    trainSize = totalSize * (1 - testRatio)
    trainSize = int(trainSize - trainSize % roundBy)
    return dataSet[:trainSize], dataSet[trainSize:]


def train_val_split(fold, dataSet):
    resDict = {}
    if len(dataSet) % fold != 0:
        raise Exception("num_data_points is not divisible by fold_k")
    splitList = np.split(dataSet, fold)
    for i, val in enumerate(splitList):
        trainSet = np.concatenate((splitList[:i] + splitList[i + 1:]))
        valSet = splitList[i]
        resDict[i] = (trainSet, valSet)
    return resDict


def split_cv_train_val_test(data, testRatio, cvFold):
    dataFilename = f"{cvFold}_fold_cv_data_split.npz"
    if os.path.exists(dataFilename):
        f = np.load(dataFilename)
        trainValDataDict = {}
        for i in range(f["keysNum"]):
            trainValDataDict[i] = (f["train" + str(i)], f["val" + str(i)])
        test = f["test"]
    else:
        train, test = train_test_split(data, testRatio=testRatio, roundBy=cvFold)
        trainValDataDict = train_val_split(fold, train)
        dictKeysNum = len(trainValDataDict.keys())
        np.savez(dataFilename, keysNum=dictKeysNum,
                 **{"train" + str(k): v[0] for k, v in trainValDataDict.items()},
                 **{"val" + str(k): v[1] for k, v in trainValDataDict.items()},
                 test=test)
    return trainValDataDict, test


def gen_model_snapshot_filename(lambda_, fold, iterCnt):
    filename = f"lambda_{float(lambda_)}_fold_{fold}_{iterCnt + 1}th.npz"
    return filename


def split_x_y(*args):
    res = []
    for dataSets in args:
        res.append(dataSets[:, 2:])
        res.append(dataSets[:, :2])
    return res


def best_model_test(bestModel, x_test, y_test):
    def _mse_mae(x, y, coef):
        x = np.array(x)
        y = np.array(y)
        coef = np.array(coef)
        y_hat = x @ coef
        mse = np.mean(np.square(y_hat - y), axis=0)
        mae = np.mean(np.abs(y_hat - y), axis=0)
        return mse, mae

    cd_coef = pd.read_csv('lasso_multi.csv').iloc[:, 1:]
    cd_coef = np.array(cd_coef)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    gd_coef = bestModel['coefficient']

    mse_train_gd = bestModel['training_mse_history'][-1]
    mae_train_gd = bestModel['training_mae_history'][-1]
    mse_test_gd, mae_test_gd = _mse_mae(x_test, y_test, gd_coef)

    mse_train_cd, mae_train_cd = _mse_mae(x_train, y_train, cd_coef)
    mse_test_cd, mae_test_cd = _mse_mae(x_test, y_test, cd_coef)

    print("---------Best Model - ISTA-----------")
    mse_table = pd.DataFrame(data=[mse_train_gd, mse_test_gd], index=['Training MSE', 'Testing MSE'],
                             columns=['y1', 'y2'])
    print(mse_table)
    mae_table = pd.DataFrame(data=[mae_train_gd, mae_test_gd], index=['Training MAE', 'Testing MAE'],
                             columns=['y1', 'y2'])
    print(mae_table)
    print("----------------CD-------------------")
    mse_table = pd.DataFrame(data=[mse_train_cd, mse_test_cd], index=['Training MSE', 'Testing MSE'],
                             columns=['y1', 'y2'])
    print(mse_table)
    mae_table = pd.DataFrame(data=[mae_train_cd, mae_test_cd], index=['Training MAE', 'Testing MAE'],
                             columns=['y1', 'y2'])
    print(mae_table)


if __name__ == "__main__":
    restartCnt = 0 if len(sys.argv) == 1 else int(sys.argv[-1])
    restartLimit = 100
    if restartCnt > 0:
        print(f"\n\n\n\n({restartCnt}/{restartLimit})th restart\n--------", file=sys.stderr)
    if restartCnt > restartLimit:
        raise Exception("Restart limit exceeded!")
    os.environ["CUDA_VISIBLE_DEVICES"] = '3'
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = '.25'
    # double
    os.environ["JAX_ENABLE_X64"] = 'True'
    from jax.config import config
    config.update("jax_enable_x64", True)

    fold = 4
    lambdaList = list([0.1,5,10,15,20,30,40,70])
    lambdaList.sort()

    data = np.loadtxt('micedata.csv', dtype=float, delimiter=',')
    data = normalize(data)
    trainValDataDict, test = split_cv_train_val_test(data, testRatio=0.1, cvFold=fold)
    for lambda_ in lambdaList:
        for i, (train, val) in trainValDataDict.items():
            x_train, y_train, x_val, y_val = train[:, 2:], train[:, :2], val[:, 2:], val[:, :2]
            try:
                model = MultiLassoIstaLineSearch(x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val,
                                                 lambda_=lambda_,
                                                 snapshot=gen_model_snapshot_filename(lambda_, fold, i),
                                                 maxIteration=10000,
                                                 verbose=True)
                print(f"\nmodel({gen_model_snapshot_filename(lambda_, fold, i)}) starts training.\n")
                model.train()
                model.save_snapshot()
            except ModelExistsException:
                print(f"model({gen_model_snapshot_filename(lambda_, fold, i)}) has already been trained.")
            except MemoryExceedsException as e:
                print(e.message)
                print(f"Memory limit exceeds. Suicide now!")
                if 0 == os.fork():
                    # kill -9 $(ps -aux|grep cv.py|grep -v grep|cut -f2 -d' ')
                    print(f"Trying to restart...")
                    proc = psutil.Process()
                    cmd = proc.cmdline()
                    cmd.append(str(restartCnt + 1))
                    os.execvp(proc.exe(), cmd)
                else:
                    sys.exit(93)

    x_axis_lambda = []
    y_axis_val_acc = []
    for lambda_ in lambdaList:
        for i in range(fold):
            modelFile = np.load(gen_model_snapshot_filename(lambda_, fold, i))
            x_axis_lambda.append(modelFile["lambda"])
            y_axis_val_acc.append(modelFile["validating_mse_history"][-1])
    x_axis_lambda = np.array(x_axis_lambda)
    y_axis_val_acc = np.mean(np.array(y_axis_val_acc), axis=1)
    assert x_axis_lambda.shape == y_axis_val_acc.shape
    x_axis_lambda_unique = np.unique(x_axis_lambda)
    y_axis_val_acc_fold_mean = np.mean(y_axis_val_acc.reshape(-1, fold), axis=1, keepdims=False)
    y_axis_val_acc_fold_std = np.std(y_axis_val_acc.reshape(-1, fold), axis=1, keepdims=False)
    assert x_axis_lambda_unique.shape == y_axis_val_acc_fold_mean.shape == y_axis_val_acc_fold_std.shape

    argmin = np.argmin(y_axis_val_acc_fold_mean)

    fig1, ax1 = plt.subplots(figsize=(8, 7))
    ax1.scatter(x_axis_lambda, y_axis_val_acc, label="val_mse")
    ax1.errorbar(x_axis_lambda_unique, y_axis_val_acc_fold_mean, yerr=y_axis_val_acc_fold_std, fmt='r^-',
                 capsize=4, label="val_mse_fold_mean")

    ax1.legend()
    ax1.grid(True, which='both', axis='both', linestyle='-.')
    ax1.set_xlabel("$\lambda$")
    # ax1.set_xscale("log")
    ax1.set_ylabel("mse")
    ax1.vlines(x_axis_lambda_unique[argmin], *ax1.get_ylim())
    ax1.annotate('minimum', xy=(x_axis_lambda_unique[argmin], y_axis_val_acc_fold_mean[argmin]),
                 xytext=(x_axis_lambda_unique[argmin], y_axis_val_acc_fold_mean[argmin] * 2))

    fig1.suptitle(f'{fold} fold cross validation for $\lambda$')
    fig1.savefig("cv_res.png")

    # All data retrain and test
    all_train = trainValDataDict[0]
    all_train = np.concatenate((all_train[0], all_train[1]))
    x_train, y_train = split_x_y(all_train)
    x_test, y_test = split_x_y(test)
    best_lambda = x_axis_lambda_unique[np.argmin(y_axis_val_acc_fold_mean)]
    bestModelFilename = f"best_model_lambda_{best_lambda}.npz"
    try:
        print(f"\n\nBest model(lambda={best_lambda}) starts training.\n")
        bestModel = MultiLassoIstaLineSearch(x_train=x_train, y_train=y_train, x_val=x_test, y_val=y_test,  # 记得删除x_test
                                             lambda_=40, #手动训练
                                             snapshot=bestModelFilename,
                                             maxIteration=1e6,
                                             verbose=True)
        bestModel.train(print_every=1000)
        bestModel.save_snapshot()
    except ModelExistsException:
        print(f"model({bestModelFilename}) has already been trained.")

    best_model_test(np.load(bestModelFilename), x_test, y_test)

    # Sample and draw lr_iter plot & loss_iter
    sampleSize = 10
    sampleSize = sampleSize if len(lambdaList) >= sampleSize else len(lambdaList)
    sampledLambda = np.sort(np.random.choice(lambdaList, sampleSize, replace=False))
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    for lambda_ in sampledLambda:
        iterCnt = random.randrange(fold)
        model = np.load(gen_model_snapshot_filename(lambda_, fold, iterCnt))
        ax1.scatter(range(1, model['iterations'] + 1), model['learning_rate_history'], s=10,
                    label=f"$\lambda={model['lambda']}$")
        ax2.plot(range(1, model['iterations'] + 1), model['loss_history'], label=f"$\lambda={model['lambda']}$")

    ax1.set_xlabel("epochs")
    ax1.set_ylabel("learning rate")
    ax1.legend()
    ax1.set_title(f"Learning rate - epochs across different $\lambda$s")
    fig1.savefig("lr_iter_samples.png")

    ax2.set_xlabel("epochs")
    ax2.set_ylabel("loss")
    ax2.legend()
    ax2.set_title(f"Loss - epochs across different $\lambda$s")
    fig2.savefig("lr_loss_samples.png")

    # Best lr-graph for best model (for debugging use)
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    model = np.load(bestModelFilename)
    ax1.scatter(range(1, model['iterations'] + 1), model['learning_rate_history'])
    ax1.set_xlabel("epochs")
    ax1.set_ylabel("learning rate")
    ax1.set_yscale("log")
    ax1.set_title(f"Learning rate - epochs for $\lambda={model['lambda']}$")
    fig1.savefig("lr.png")

    # Coefficient graph across different lambdas
    selectedNum = 6
    selectedCoefIndex = np.sort(np.random.choice(range(x_test.shape[1]), selectedNum, replace=False))
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    coefList = np.zeros(shape=(selectedNum, len(lambdaList)))
    for j, lambda_ in enumerate(lambdaList):
        coefSum = np.zeros(selectedNum)
        for iterCnt in range(fold):
            model = np.load(gen_model_snapshot_filename(lambda_, fold, iterCnt))
            coefSum += model['coefficient'][selectedCoefIndex][:, 0]
        coefMean = coefSum / fold
        coefList[:, j] = coefMean
    for idx, coef_series in enumerate(coefList):
        ax1.plot(lambdaList, coef_series, label=f"y1_coef #{selectedCoefIndex[idx]}")
    ax1.set_xlabel("Lambdas")
    ax1.set_xscale("log")
    ax1.set_ylabel("Coef")
    ax1.set_title(f"Coef across different lambdas$")
    fig1.savefig("coef_lambdas.png")

    # coef_sum-iter graph
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    model = np.load(bestModelFilename)
    ax1.scatter(list(range(1, model['iterations'] + 1))[14000:], model['coef_sum_history'][14000:], s=10)
    ax1.set_xlabel("epochs")
    ax1.set_ylabel("|coefficient| sum")
    ax1.set_title(f"|Coefficient| sum - epochs for $\lambda={model['lambda']}$")
    fig1.savefig("coef_sum.png")

    # delta_coef_norm-iter graph
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    model = np.load(bestModelFilename)
    ax1.scatter(list(range(1, model['iterations'] + 1))[6000:], model['delta_coef_norm_history'][6000:], s=10)
    ax1.set_xlabel("epochs")
    ax1.set_ylabel("Delta_coefficient_norm")
    ax1.set_title(f"Delta_coefficient - epochs for $\lambda={model['lambda']}$")
    fig1.savefig("delta_coef_norm.png")

    # loss-iter graph
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    model = np.load(bestModelFilename)
    ax1.scatter(list(range(1, model['iterations'] + 1)), model['loss_history'], s=10)
    ax1.set_xlabel("epochs")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"Loss - epochs for $\lambda={model['lambda']}$")
    fig1.savefig("loss.png")

    # debug_test_mse-iter graph
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    model = np.load(bestModelFilename)
    ax1.scatter(list(range(1, model['iterations'] + 1)), model['validating_mse_history'][:, 0], s=5)
    ax1.scatter(list(range(1, model['iterations'] + 1)), model['validating_mse_history'][:, 1], s=5)
    ax1.set_xlabel("epochs")
    ax1.set_ylabel("Test MSE")
    ax1.set_title(f"Test MSE - epochs for $\lambda={model['lambda']}$")
    fig1.savefig("debug_test_mse.png")

    # nonzero_coef_number-iter graph
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    model = np.load(bestModelFilename)
    ax1.scatter(list(range(1, model['iterations'] + 1)), model['nonzero_coef_num'], s=5)
    ax1.set_xlabel("epochs")
    ax1.set_ylabel("Number of nonzero coefficient")
    ax1.set_yscale("log")
    ax1.set_title(f"Number of nonzero coefficient - epochs for $\lambda={model['lambda']}$")
    fig1.savefig("nonzero_coef_num.png")

    # nonzero_coef_spy graph
    fig1, ax1 = plt.subplots(figsize=(3, 18))
    model = np.load(bestModelFilename)
    ax1.spy(model['coefficient'].reshape(739,-1))
    ax1.set_title(f"Coeffient distribution for $\lambda={model['lambda']}$")
    fig1.savefig("nonzero_coef_spy.png")
