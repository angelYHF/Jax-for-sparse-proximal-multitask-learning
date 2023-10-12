import argparse
import codecs
import matplotlib.pyplot as plt

import tools
from tools import normalize, split_x_y, mse
from dataset import load_csv_data
import repo as model_manager
from trainer import MultiProcessModelTrainer

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--override', action='store_true',
                        help='Override started but unfinished checkpoints.')
    args = parser.parse_args()

    repoName = 'admm'
    tools.switch_model(repoName)
    from model import MixedLassoISTA

    data = load_csv_data('micedata.csv')
    data.data = normalize(data.data)
    train, test = data.split(testRatio=0.1)

    x_train, y_train = split_x_y(train)
    x_test, y_test = split_x_y(test)

    try:
        repo = model_manager.open_repository(repoName)
    except model_manager.RepositoryNotExists:
        repo = model_manager.create_repository(repoName=repoName, identityFields=['lambda1', 'lambda2', 'e'])

    lambda1 = 13.02
    lambda2 = 1.26
    maxIterationNum = 10000
    e = 1e-4

    model = MixedLassoISTA(
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
    trainer = MultiProcessModelTrainer(model)
    trainer.train(print_every=10)

    # loss-epoch图
    fig, ax = plt.subplots(figsize=(10, 10))
    model = repo.pick(lambda1=lambda1, lambda2=lambda2, e=e)
    ax.plot(range(1, model['iterations'] + 1), model['loss_history'],
            label=f"$lambdas={(float(model['lambda1']), float(model['lambda2']))}$")

    ax.set_xlabel("epochs")
    ax.set_ylabel("loss")
    ax.legend()
    ax.set_yscale("log")
    ax.set_title(f"Loss - Epoch")
    fig.savefig(f"{repoName}_loss_epoch.png")

    with codecs.open(f'{repoName}_report.txt', 'a', 'utf8') as f:
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
