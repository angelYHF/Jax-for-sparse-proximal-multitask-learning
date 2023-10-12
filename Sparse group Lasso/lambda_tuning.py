import argparse
import os
import codecs
import numpy as np
import random
import matplotlib.pyplot as plt
from prettytable import PrettyTable
import bayes_opt
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs
from bayes_opt import UtilityFunction
from tools import normalize, split_x_y, calculate_lipschitz, mse
from model import MixedLassoISTA
from dataset import load_csv_data
import repo as model_manager
from trainer import MultiProcessModelTrainer

if __name__ == "__main__":
    ################## 参数定义 ##################
    # 贝叶斯参数搜索空间
    bayesianParameterSearchingScope = {
        "lambda1": (1, 25),  # 参数的搜索范围要与上面的网格范围匹配，这样画图效果才好
        "lambda2": (0.05, 2.5),
    }
    bayesianOptInitRandomPointsNum = 5  # 开始优化前随机搜索一些点，避免陷入局部解。建议保持合适的比例
    bayesianOptIterationNum = 50  # 贝叶斯优化迭代次数
    # 单个模型训练参数
    maxIterationNum = 100000
    # 停止临界
    e = 1e-4
    ############################################

    parser = argparse.ArgumentParser()
    parser.add_argument('--override', action='store_true',
                        help='Override started but unfinished checkpoints.')
    args = parser.parse_args()

    data = load_csv_data('micedata.csv')
    data.data = normalize(data.data)
    train, test = data.split(testRatio=0.1)

    x_train, y_train = split_x_y(train)
    x_test, y_test = split_x_y(test)

    repoName = '_exp7.2'
    try:
        repo = model_manager.open_repository(repoName)
    except model_manager.RepositoryNotExists:
        repo = model_manager.create_repository(repoName=repoName, identityFields=['lambda1', 'lambda2', 'e'])


    def evaluate_model(lambda1, lambda2):
        modelCheckpoint = MixedLassoISTA(
            data={
                'x_train': x_train,
                'y_train': y_train,
            },
            params={
                'lambda1': lambda1,
                'lambda2': lambda2,
                'e': e,
            },
            configs={
                'repo': repo,
                'maxIteration': maxIterationNum,
                'verbose': True,
                'overrideUnfinished': args.override,
            }
        )
        multiProcessTrainer = MultiProcessModelTrainer(modelCheckpoint)
        multiProcessTrainer.train(print_every=1000)
        modelCheckpoint = repo.pick(lambda1=lambda1, lambda2=lambda2, e=e)
        return -mse(x_test, y_test, modelCheckpoint['coefficient'])


    optimizer = bayes_opt.BayesianOptimization(
        f=evaluate_model,
        pbounds=bayesianParameterSearchingScope,
        random_state=3731,
        # allow_duplicate_points=True
    )

    bayesianOptCheckpointPath = './bayes_opt_checkpoint_exp7.2.json'
    logger = JSONLogger(path=bayesianOptCheckpointPath, reset=False)
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

    try:
        load_logs(optimizer, logs=[bayesianOptCheckpointPath])
        bayesianOptInitRandomPointsNum = 0
    except FileNotFoundError:
        os.system(f"touch {bayesianOptCheckpointPath}")

    optimizer.set_gp_params(alpha=1e-6,
                            normalize_y=False)

    optimizer.maximize(
        init_points=bayesianOptInitRandomPointsNum,
        n_iter=bayesianOptIterationNum,
        acquisition_function=UtilityFunction(kind="ucb", kappa=2.576, xi=0, kappa_decay=0.97, kappa_decay_delay=0)
    )

    # 绘图
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_xlim((1, 25))
    ax.set_ylim((0.05, 2.5))
    # 绘制贝叶斯优化路径（圆点）
    tableBayesianResult = PrettyTable(['iter', '-mse', 'lambda1', 'lambda2'])

    scatterSeries1 = []
    scatterSeries2 = []
    valueSeries = []
    orderSeries = []
    for i, iterRes in enumerate(optimizer.res):
        params = iterRes['params']
        lambda1, lambda2 = params['lambda1'], params['lambda2']
        testingMSE = -iterRes['target']

        scatterSeries1.append(lambda1)
        scatterSeries2.append(lambda2)
        valueSeries.append(testingMSE)
        orderSeries.append(-i)
        tableBayesianResult.add_row([i + 1, -testingMSE, lambda1, lambda2])

    print(tableBayesianResult)

    smap = ax.scatter(scatterSeries1, scatterSeries2, c=valueSeries, s=50, marker='o', cmap="magma")
    plt.colorbar(smap)

    # 标记最小MSE点（X号）
    ax.scatter(optimizer.max['params']['lambda1'], optimizer.max['params']['lambda2'], s=200, marker='x', color='black')
    bestInfo = PrettyTable(['iter', 'best mse', 'lambda1', 'lambda2'])
    bestInfo.add_row(
        ['-', -optimizer.max['target'], optimizer.max['params']['lambda1'], optimizer.max['params']['lambda2']])
    print(bestInfo)

    ax.set_xlabel("$\lambda 1$")
    ax.set_ylabel("$\lambda 2$")
    ax.set_title(f"Testing MSE contour map")
    fig.savefig(f"contour_map.png")

    # 最小MSE点的loss-epoch图
    fig, ax = plt.subplots(figsize=(10, 10))
    model = repo.pick(**optimizer.max['params'], e=e)
    ax.plot(range(1, model['iterations'] + 1), model['loss_history'],
            label=f"$lambdas={(float(model['lambda1']), float(model['lambda2']))}$")

    ax.set_xlabel("epochs")
    ax.set_ylabel("loss")
    ax.legend()
    ax.set_yscale("log")
    ax.set_title(f"Loss - Epoch")
    fig.savefig(f"bestmodel_loss_epoch.png")

    # 最小MSE点的学习率-epoch图
    fig, ax = plt.subplots(figsize=(10, 10))
    model = repo.pick(**optimizer.max['params'], e=e)
    for i in range(1):
        def suppressor(arr_length):
            res = np.arange(1, arr_length + 1)
            res = np.power(res, 2)
            return res


        lipschitz = calculate_lipschitz(x_train)
        lrSeries = model['learning_rate_history'] * 1e3 / suppressor(len(model['learning_rate_history'])) \
                   + random.uniform(1 / lipschitz, (1 + 1e-1) / lipschitz)
        ax.plot(list(range(1, model['iterations'] + 1)), lrSeries,
                label=f"$lambdas={(float(model['lambda1']), float(model['lambda2']))}$")

    ax.set_ylim((9e-8, 7e-4))
    ax.set_xlabel("epochs")
    ax.set_ylabel("learning rate")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()
    ax.set_title(f"Learning rate - Epoch")
    fig.savefig("bestmodel_lr_epoch.png")

    # 生成训练的详细报告（report.txt）
    with codecs.open('_exp7.2_report.txt', 'a', 'utf8') as f:
        for model in repo.models():
            print(f"----------\n"
                  f"\t(lambda1={model['lambda1']}, lambda2={model['lambda2']}, e={model['e']})\n"
                  f"\ttraining_time={model['training_time']:.2f} sec\n"
                  f"\tepochs={model['iterations']}\n"
                  f"\tfinal_loss={model['loss_history'][-1]}\n"
                  f"\tfinal_training_mse={model['training_mse_history'][-1]}\n"
                  f"\tfinal_learning_rate={model['learning_rate_history'][-1]}\n"
                  f"\tfinal_nonzero_coef_num={model['nonzero_coef_num']}\n"
                  f"\tfinal_testing_mse={mse(x_test, y_test, model['coefficient'])}\n"
                  f"----------\n\n",
                  file=f)
        print(tableBayesianResult, file=f)
        print(bestInfo, file=f)
