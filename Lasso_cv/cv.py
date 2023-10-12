import os
import random
import sys
import psutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lib.ista import MultiLassoIstaLineSearch, ModelExistsException, MemoryExceedsException

# 0-1标准化
def normalize(numpy_ndarray):
    mu = np.mean(numpy_ndarray)
    std = np.std(numpy_ndarray)
    return (numpy_ndarray - mu) / std

# 切分训练、测试集
def train_test_split(dataSet, testRatio, roundBy):
    np.random.default_rng().shuffle(dataSet, axis=0)
    totalSize = len(dataSet)
    trainSize = totalSize * (1 - testRatio)
    trainSize = int(trainSize - trainSize % roundBy)
    return dataSet[:trainSize], dataSet[trainSize:]

# 切分训练、验证集
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

# 切分训练、验证、测试集
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
    filename = f"lambda_{lambda_}_fold_{fold}_{iterCnt + 1}th.npz"
    return filename

# 切分x、y标签
def split_x_y(*args):
    res = []
    for dataSets in args:
        res.append(dataSets[:, 2:])
        res.append(dataSets[:, :2])
    return res

# 最佳模型测试，并与sklearn CD法训练出来的结果对比
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
    # 设置重启参数
    restartCnt = 0 if len(sys.argv) == 1 else int(sys.argv[-1])
    restartLimit = 100
    if restartCnt > 0:
        print(f"\n\n\n\n({restartCnt}/{restartLimit})th restart\n--------", file=sys.stderr)
    if restartCnt > restartLimit:
        raise Exception("Restart limit exceeded!")
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = '.25'
    os.environ["JAX_ENABLE_X64"] = 'True'

    # 注意：在这里设置k-fold验证的k
    fold = 10 # 10折验证
    # 注意：设置要验证的lambda，放进下面的列表里
    lambdaList = list({9000, 800, 700})
    lambdaList.sort()

    # 开始cross validation
    data = np.loadtxt('micedata.csv', dtype=float, delimiter=',')
    data = normalize(data)
    # 抽取数据
    trainValDataDict, test = split_cv_train_val_test(data, testRatio=0.1, cvFold=fold)
    for lambda_ in lambdaList:
        for i, (train, val) in trainValDataDict.items():
            x_train, y_train, x_val, y_val = train[:, 2:], train[:, :2], val[:, 2:], val[:, :2]
            try:
                model = MultiLassoIstaLineSearch(x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val,
                                                 lambda_=lambda_,
                                                 snapshot=gen_model_snapshot_filename(lambda_, fold, i),
                                                 maxIteration=2000,
                                                 verbose=True)
                print(f"\nmodel({gen_model_snapshot_filename(lambda_, fold, i)}) starts training.\n")
                model.train()
                model.save_snapshot()
            except ModelExistsException:
                print(f"model({gen_model_snapshot_filename(lambda_, fold, i)}) has already been trained.")
            except MemoryExceedsException as e:
                # 内存超出限制，自动重启
                print(e.message)
                print(f"Memory limit exceeds. Suicides now!")
                if 0 == os.fork():
                    # 如果无法结束重启的过程，用下面的命令杀死进程
                    # kill -9 $(ps -aux|grep cv.py|grep -v grep|cut -f2 -d' ')
                    print(f"Trying to restart...")
                    proc = psutil.Process()
                    cmd = proc.cmdline()
                    cmd.append(str(restartCnt + 1))
                    os.execvp(proc.exe(), cmd)
                else:
                    sys.exit(93)

    # 开始绘制调参图
    x_axis_lambda = []
    y_axis_val_acc = []
    for lambda_ in lambdaList:
        for i in range(fold):
            modelFile = np.load(gen_model_snapshot_filename(lambda_, fold, i))
            x_axis_lambda.append(modelFile["lambda"])
            y_axis_val_acc.append(modelFile["validating_mse_history"][-1])
    x_axis_lambda = np.array(x_axis_lambda)
    y_axis_val_acc = np.array(y_axis_val_acc)
    x_axis_lambda_unique = np.unique(x_axis_lambda)
    y_axis_val_acc_fold_mean = np.mean(y_axis_val_acc.reshape(-1, fold, 2), axis=1, keepdims=False)
    y_axis_val_acc_fold_std = np.std(y_axis_val_acc.reshape(-1, fold, 2), axis=1, keepdims=False)

    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7))
    fig1.subplots_adjust(hspace=0.2)

    ax1.scatter(x_axis_lambda, y_axis_val_acc[:, 0], label="y1_val_mse")
    ax1.errorbar(x_axis_lambda_unique, y_axis_val_acc_fold_mean[:, 0], yerr=y_axis_val_acc_fold_std[:, 0], fmt='^-',
                 capsize=4, label="y1_val_mse_fold_mean")

    ax1.legend()
    ax1.set_xlabel("$\lambda$")
    ax1.set_xscale("log")
    ax1.set_ylabel("mse")

    ax2.scatter(x_axis_lambda, y_axis_val_acc[:, 1], label="y2_val_mse")
    ax2.errorbar(x_axis_lambda_unique, y_axis_val_acc_fold_mean[:, 1], yerr=y_axis_val_acc_fold_std[:, 1], fmt='^-',
                 capsize=4, label="y2_val_mse_fold_mean")

    ax2.legend()
    ax2.set_xlabel("$\lambda$")
    ax2.set_xscale("log")
    ax2.set_ylabel("mse")
    fig1.suptitle(f'{fold} fold cross validation for $\lambda$')

    fig1.savefig("cv_res.png")

    # 重新生成全部的训练集来训练最佳模型
    all_train = trainValDataDict[0]
    all_train = np.concatenate((all_train[0], all_train[1]))
    x_train, y_train = split_x_y(all_train)
    x_test, y_test = split_x_y(test)
    best_lambda = x_axis_lambda_unique[np.argmin(y_axis_val_acc_fold_mean[:, 0])]
    bestModelFilename = f"best_model_lambda_{best_lambda}.npz"
    try:
        print(f"\n\nBest model(lambda={best_lambda}) starts training.\n")
        bestModel = MultiLassoIstaLineSearch(x_train=x_train, y_train=y_train, x_val=None, y_val=None,
                                             lambda_=best_lambda,
                                             snapshot=bestModelFilename,
                                             maxIteration=2000,
                                             verbose=True)
        bestModel.train()
        bestModel.save_snapshot()
    except ModelExistsException:
        print(f"model({bestModelFilename}) has already been trained.")

    best_model_test(np.load(bestModelFilename), x_test, y_test)

    # 绘制不同lambda的学习率对比曲线
    sampleSize = 10 # 设置图中要对比几个lambda
    sampleSize = sampleSize if len(lambdaList) >= sampleSize else len(lambdaList)
    sampledLambda = np.sort(np.random.choice(lambdaList, sampleSize, replace=False)) # 随机抽取lambda
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    for lambda_ in sampledLambda:
        iterCnt = random.randrange(fold)
        model = np.load(gen_model_snapshot_filename(lambda_, fold, iterCnt))
        ax1.scatter(range(1, model['iterations'] + 1), model['learning_rate_history'], s=10,
                    label=f"$\lambda={model['lambda']}$")

    ax1.set_xlabel("epochs")
    ax1.set_ylabel("learning rate")
    ax1.set_yscale("log")
    ax1.legend()
    ax1.set_title(f"Learning rate - epochs across different $\lambda$s")
    fig1.savefig("lr_iter_samples.png")
