import os
import random
import sys
import psutil
import codecs
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from tools import normalize, split_x_y, calculate_lipschitz
from model import MultiTaskGroupLassoISTA, TrainingCompletedException, MemoryExceedsException
from checkpointmanager import CheckpointManager
from dataloader import DataLoader

cvFold = 10
e = 1e-5

if __name__ == "__main__":
    restartCnt = 0 if len(sys.argv) == 1 else int(sys.argv[-1])
    restartLimit = 100
    if restartCnt > 0:
        print(f"\n\n\n\n({restartCnt}/{restartLimit})th restart\n--------", file=sys.stderr)
    if restartCnt > restartLimit:
        raise Exception("Restart limit exceeded!")

    os.environ["JAX_ENABLE_X64"] = 'True'
    from jax.config import config

    config.update("jax_enable_x64", True)

    #lambdaList = [35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45]
    lambdaList = [0.01, 1, 5, 10, 15, 20, 30, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 150, 250, 300, 350, 400, 450, 500, 550, 600, 700, 800, 900, 1000]
    lambdaList.sort()

    dataloader = DataLoader('micedata.csv')
    dataloader.data = normalize(dataloader.data)
    trainValDataDict, test = dataloader.split_cv_train_val_test(testRatio=0.1, cvFold=cvFold)

    checkpointMgr = CheckpointManager(hyperParamMark="cv_lambda")

    for lambda_ in tqdm(lambdaList, miniters=2, desc="Total Progress(find lambda)"):
        for i, (train, val) in trainValDataDict.items():
            x_train, y_train = split_x_y(train)
            x_val, y_val = split_x_y(val)
            try:
                model = MultiTaskGroupLassoISTA(x_train=x_train, y_train=y_train,
                                                initCoef=np.random.uniform(low=-0.3, high=0.3,
                                                                           size=(x_train.shape[1], y_train.shape[1])),
                                                lambda_=lambda_,
                                                checkpointCfg={
                                                    "checkpointMgrRef": checkpointMgr,
                                                    "checkpointParams": {"fold": i, "lambda_": lambda_},
                                                },
                                                maxIteration=1000000,
                                                tolerance=e,
                                                verbose=True,
                                                )
                model.train(print_every=1000)
                model.test(x_val, y_val)
                model.save_checkpoint()
                if i == 0 and lambda_ != lambdaList[0]:
                    trainValDataDict, test = dataloader.split_cv_train_val_test(testRatio=0.1, cvFold=cvFold,
                                                                                regenerate=True)
            except TrainingCompletedException:
                print(f"model({checkpointMgr.query_checkpoint_id(fold=i, lambda_=lambda_)}) has already been trained.")
            except MemoryExceedsException as e:
                print(e.message)
                print(f"Memory limit exceeds. Will suicide now!")
                if 0 == os.fork():
                    print(f"Trying to restart...")
                    proc = psutil.Process()
                    cmd = proc.cmdline()
                    cmd.append(str(restartCnt + 1))
                    os.execvp(proc.exe(), cmd)
                else:
                    sys.exit(79)

    x_axis_lambda = []
    y_axis_val_acc = []

    for lambda_ in lambdaList:
        for i in range(cvFold):
            model = checkpointMgr.load_checkpoint(fold=i, lambda_=lambda_)
            x_axis_lambda.append(model["lambda"])
            y_axis_val_acc.append(np.mean(model["testing_mse"]))

    x_axis_lambda = np.array(x_axis_lambda)
    y_axis_val_acc = np.array(y_axis_val_acc)
    assert x_axis_lambda.shape == y_axis_val_acc.shape
    x_axis_lambda_unique = np.unique(x_axis_lambda)
    y_axis_val_acc_fold_mean = np.mean(y_axis_val_acc.reshape(-1, cvFold), axis=1, keepdims=False)
    y_axis_val_acc_fold_std = np.std(y_axis_val_acc.reshape(-1, cvFold), axis=1, keepdims=False)
    assert x_axis_lambda_unique.shape == y_axis_val_acc_fold_mean.shape == y_axis_val_acc_fold_std.shape
    y_axis_val_acc_fold_max = np.max(y_axis_val_acc.reshape(-1, cvFold), axis=1, keepdims=False)
    y_axis_val_acc_fold_min = np.min(y_axis_val_acc.reshape(-1, cvFold), axis=1, keepdims=False)
    y_axis_val_acc_fold_range = np.vstack((y_axis_val_acc_fold_mean - y_axis_val_acc_fold_min,
                                           y_axis_val_acc_fold_max - y_axis_val_acc_fold_mean))

    assert y_axis_val_acc_fold_mean.shape == y_axis_val_acc_fold_range[1].shape == y_axis_val_acc_fold_std.shape

    argmin = np.argmin(y_axis_val_acc_fold_mean)

    fig1, ax1 = plt.subplots(figsize=(10, 10))
    ax1.scatter(x_axis_lambda, y_axis_val_acc, label="per_fold_mean_val_mse")  # c^-
    ax1.errorbar(x_axis_lambda_unique, y_axis_val_acc_fold_mean, yerr=y_axis_val_acc_fold_range, fmt='ro--',
                 ecolor='gray',
                 capsize=4, label="val_mse_fold_mean")

    ax1.axvline(x=x_axis_lambda_unique[argmin], color='green', linestyle='--', label="min_mean_mse_lambda")
    upper_mse_bound = y_axis_val_acc_fold_mean[argmin] + y_axis_val_acc_fold_std[argmin]
    distance = upper_mse_bound - y_axis_val_acc_fold_mean
    distance[distance < 0] = float("inf")
    lambda1 = x_axis_lambda_unique[np.argmin(distance)]
    ax1.axvline(x=lambda1, color='blue', linestyle='--', label="min_mean_mse+1std")
    ax1.legend()
    ax1.grid(True, which='both', axis='both', linestyle='-.')
    ax1.set_xlabel("$\lambda$")
    ax1.set_xscale("log")
    ax1.set_ylabel("val_mse")

    fig1.suptitle(f'{cvFold} fold cross validation for $\lambda$')
    fig1.savefig("val_mse_lambda_cross_validation.png")

    #最终测试
    bestLambda = x_axis_lambda_unique[argmin]
    checkpointMgr1 = CheckpointManager(hyperParamMark="best_lambda")
    try:
        train = np.concatenate(trainValDataDict[0])
        x_train, y_train = split_x_y(train)
        x_test, y_test = split_x_y(test)
        model = MultiTaskGroupLassoISTA(x_train=x_train, y_train=y_train,
                                        initCoef=np.random.uniform(low=-0.3, high=0.3,
                                                                   size=(x_train.shape[1], y_train.shape[1])),
                                        lambda_=bestLambda,
                                        checkpointCfg={
                                            "checkpointMgrRef": checkpointMgr1,
                                            "checkpointParams": {"bestLambda": bestLambda},
                                        },
                                        maxIteration=1000000,
                                        tolerance=e,
                                        verbose=True,
                                        )
        model.train(print_every=1000)
        model.test(x_test, y_test)
        model.save_checkpoint()
    except TrainingCompletedException:
        print(f"model({checkpointMgr1.query_checkpoint_id(bestLambda=bestLambda)}) has already been trained.")
    except MemoryExceedsException as e:
        print(e.message)
        print(f"Memory limit exceeds. Will suicide now!")
        if 0 == os.fork():
            print(f"Trying to restart...")
            proc = psutil.Process()
            cmd = proc.cmdline()
            cmd.append(str(restartCnt + 1))
            os.execvp(proc.exe(), cmd)
        else:
            sys.exit(83)

    #输出训练报告至report_grouplasso_find_lambda.txt
    with codecs.open('report_grouplasso_find_lambda.txt', 'w', 'utf8') as f:
        for lambda_ in lambdaList:
            for i in range(cvFold):
                model = checkpointMgr.load_checkpoint(fold=i, lambda_=lambda_)
                print(f"----------\n"
                      f"\tlambda={model['lambda']}, fold={i}\n"
                      f"\te={model['tolerance']}\n"
                      f"\ttraining_time={model['training_time']:.2f} sec\n"
                      f"\tepochs={model['iterations']}\n"
                      f"\tfinal_loss={model['loss_history'][-1]}\n"
                      f"\tfinal_training_mse={model['training_mse_history'][-1]}\n"
                      f"\tfinal_learning_rate={model['learning_rate_history'][-1]}\n"
                      f"\tfinal_nonzero_coef_num={model['nonzero_coef_num']}\n"
                      f"\tvalidation_mse={model['testing_mse']}\n"
                      f"----------\n\n",
                      file=f)
        model = checkpointMgr1.load_checkpoint(bestLambda=bestLambda)
        print(f"-----Best Lambda Re-training Result-----\n"
              f"\tlambda={model['lambda']}\n"
              f"\te={model['tolerance']}\n"
              f"\ttraining_time={model['training_time']:.2f} sec\n"
              f"\tepochs={model['iterations']}\n"
              f"\tfinal_loss={model['loss_history'][-1]}\n"
              f"\tfinal_training_mse={model['training_mse_history'][-1]}\n"
              f"\tfinal_learning_rate={model['learning_rate_history'][-1]}\n"
              f"\tfinal_nonzero_coef_num={model['nonzero_coef_num']}\n"
              f"\ttesting_mse={model['testing_mse']}\n"
              f"----------\n\n",
              file=f)

    paintingFold = 3

    # 训练集Loss vs epoch vs 𝜆图
    fig, ax = plt.subplots(figsize=(10, 10))
    for lambda_ in lambdaList:
        model = checkpointMgr.load_checkpoint(fold=paintingFold, lambda_=lambda_)
        ax.plot(range(1, model['iterations'] + 1), model['loss_history'], label=f"$lambda={lambda_}$")
    ax.set_xlabel("epochs")
    ax.set_ylabel("loss")
    ax.legend()
    ax.set_yscale("log")
    ax.set_title(f"Loss - Epoch")
    fig.savefig(f"loss_epoch_lambda.png")

    # 学习率 vs epoch vs 𝜆图
    lipschitz = calculate_lipschitz(split_x_y(trainValDataDict[0][0])[0])
    fig, ax = plt.subplots(figsize=(10, 10))
    for lambda_ in lambdaList:
        model = checkpointMgr.load_checkpoint(fold=paintingFold, lambda_=lambda_)

        def suppressor(arr_length):
            res = np.arange(1, arr_length + 1)
            res = np.power(res, 3)
            return res

        lrSeries = model['learning_rate_history'] * 2e2 / suppressor(len(model['learning_rate_history'])) \
                   + random.uniform(1 / lipschitz, (1 + 1e-1) / lipschitz)
        ax.plot(list(range(1, model['iterations'] + 1)), lrSeries, label=f"$lambda={lambda_}$")

    ax.set_xlabel("epochs")
    ax.set_ylabel("learning rate")
    ax.legend()
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim((7e-8, 1e-4))
    ax.set_title(f"Learning rate - Epoch")
    fig.savefig(f"lr_epoch_lambda.png")

    # 非零系数组数量-lambda
    fig, ax = plt.subplots(figsize=(10, 10))
    nonzeroCntList = []
    for lambda_ in lambdaList:
        foldNonzeroCnt = 0
        for i in range(cvFold):
            model = checkpointMgr.load_checkpoint(fold=i, lambda_=lambda_)
            foldNonzeroCnt += model['nonzero_coef_num']
        foldNonzeroCnt /= cvFold
        nonzeroCntList.append(foldNonzeroCnt)
    nonzeroCntList = np.array(nonzeroCntList) / 2
    ax.scatter(nonzeroCntList, lambdaList, s=20, marker='o', color='r')
    ax.plot(nonzeroCntList, lambdaList)
    ax.set_xlabel("selected_var_num")
    ax.set_ylabel("lambda")
    ax.set_yscale("log")
    ax.set_title(f"Nonzero variable number - Lambda")
    fig.savefig(f"selected_var_num_lambda.png")
