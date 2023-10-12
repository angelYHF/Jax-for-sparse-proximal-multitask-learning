import os
import random
import sys
import psutil
import codecs
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from tools import normalize, split_x_y
from model import MultiTaskGroupLassoISTA, TrainingCompletedException, MemoryExceedsException
from checkpointmanager import CheckpointManager
from dataloader import DataLoader

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

    e_lambda=30 #本次lambda固定为30
    eList = [1e-3, 1e-4, 1e-5, 1e-6]  #4个e值
    eList.sort()

    dataloader = DataLoader('micedata.csv')
    dataloader.data = normalize(dataloader.data)
    train, test = dataloader.split_train_test(testRatio=0.1, fixed=False)
    x_train, y_train = split_x_y(train)
    x_test, y_test = split_x_y(test)

    checkpointMgr = CheckpointManager(hyperParamMark="e_lambda")
    for e in tqdm(eList, miniters=2, desc="Total Progress(find e)"):
        try:
            model = MultiTaskGroupLassoISTA(x_train=x_train, y_train=y_train,
                                        initCoef=np.random.uniform(low=-0.3, high=0.3,
                                                                   size=(x_train.shape[1], y_train.shape[1])),
                                        lambda_=e_lambda,
                                        checkpointCfg={
                                            "checkpointMgrRef": checkpointMgr,
                                            "checkpointParams": {"e": e, "lambda_": e_lambda},
                                        },
                                        maxIteration=1000000,
                                        tolerance=e,
                                        verbose=True,
                                        )
            model.train(print_every=1000)
            model.test(x_test, y_test)
            model.save_checkpoint()
        except TrainingCompletedException:
            print(f"model({checkpointMgr.query_checkpoint_id(e=e, lambda_=e_lambda)}) has already been trained.")
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
                sys.exit(37)

    #Testing MSE vs e
    fig, ax = plt.subplots(figsize=(10, 10))
    meanMSEList = []
    for e in eList:
        model = checkpointMgr.load_checkpoint(e=e, lambda_=e_lambda)
        meanMSEList.append(np.mean(model['testing_mse']))

    ax.scatter(eList, meanMSEList, s=20, marker='o', color='r')
    ax.plot(eList, meanMSEList)
    ax.set_xlabel("e")
    ax.set_ylabel("mean testing mse")
    ax.set_xscale("log")
    ax.set_title(f"e - Mean Testing MSE")
    fig.savefig(f"e_mean_testing_mse.png")

    #输出训练报告至report_grouplasso_find_e.txt
    with codecs.open('report_grouplasso_find_e.txt', 'w', 'utf8') as f:
        for e in eList:
            model = checkpointMgr.load_checkpoint(e=e, lambda_=e_lambda)
            print(f"----------\n"
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
