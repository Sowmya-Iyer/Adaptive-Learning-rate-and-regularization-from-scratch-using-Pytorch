#
import argparse
import seaborn as sns # type: ignore
import matplotlib.pyplot as plt # type: ignore
import hashlib
import os
import numpy as onp
import torch
import pandas as pd # type: ignore
import copy
from typing import Tuple, Dict, Union, List


def load(
    *,
    seed: int, shuffle: bool, batch_size: int, cnn: bool, cgcnn: bool,
    kernel: int, stride: int, amprec: bool, optim_alg: str, lr: float,
    wd: float, ce: bool, file:str,
) -> Tuple[float, pd.DataFrame]:
    R"""
    Load log.
    """
    #
    identifier = file
    #
    stderr = os.path.join("sbatch", "{:s}.stderr.txt".format(identifier))
    ptlog = os.path.join("ptlog", "{:s}.ptlog".format(identifier))

    #
    with open(stderr, "r") as file:
        #
        for line in file:
            #
            (key, val) = line.strip().split(": ")
            if key == "Elapsed":
                #
                (val, unit) = val.split(" ")
                if unit == "sec":
                    #
                    runtime = float(val)
                else:
                    # UNEXPECT:
                    # Unknown runtime unit.
                    raise RuntimeError("Unknown runtime unit.")

    #
    data_dict: Dict[str, Union[List[float], List[int], List[str]]]

    #
    data_dict = {}
    accs = onp.array(torch.load(ptlog))
    n = len(accs)
    data_dict["Epoch"] = list(range(n))
    data_dict["Seed"] = [str(seed)] * n
    data_dict["Shuffle"] = ["Shuffle" if shuffle else "No-Shuffle"] * n
    data_dict["Batch Size"] = [str(batch_size)] * n
    data_dict["Model"] = (
        [
            {
                (False, False): "MLP", (True, False): "CNN",
                (False, True): "CGCNN",
            }
            [(cnn, cgcnn)],
        ]
        * n
    )
    data_dict["Convolve"] = [", ".join([str(kernel), str(stride)])] * n
    data_dict["AMP"] = ["AMP" if amprec else "No-AMP"] * n
    data_dict["Optim"] = [optim_alg] * n
    data_dict["LR"] = [str(lr)] * n
    data_dict["WD"] = [str(wd)] * n

    #
    if ce:
        data_dict_train_ce = copy.deepcopy(data_dict)
        data_dict_train_ce["Cross Entropy"] = accs[:, 0].tolist()
        data_dict_train_ce["Case"] = ["Train CE"] * n
        return (runtime, pd.DataFrame(data_dict_train_ce))
    else:
        #
        data_dict_train_acc = copy.deepcopy(data_dict)
        data_dict_train_acc["Accuracy"] = accs[:, 1].tolist()
        data_dict_train_acc["Case"] = ["Train Acc"] * n
        data_dict_test_acc = copy.deepcopy(data_dict)
        data_dict_test_acc["Accuracy"] = accs[:, 2].tolist()
        data_dict_test_acc["Case"] = ["Test Acc"] * n
        return (
            (
                runtime,
                pd.concat(
                    (
                        pd.DataFrame(data_dict_train_acc),
                        pd.DataFrame(data_dict_test_acc),
                    ),
                    ignore_index=True,
                ),
            )
        )
def task_amprec(ce: bool, /) -> None:
    R"""
    CNN with amprec task.
    """
    #
    buf_runtime = []
    buf_frame = []
    (runtime, frame) = (
                load(
                    seed=47, shuffle=False, batch_size=100, cnn=False,
                    cgcnn=False, kernel=5, stride=1, amprec=True,
                    optim_alg="default", lr=0.001, wd=0.0, ce=ce, file="70ab35d84a21c7c3142a33363826cee6"
                )
            )
    buf_runtime.append(runtime)
    buf_frame.append(frame)
    frame = pd.concat(buf_frame, ignore_index=True)

    #
    grids = (
        sns.relplot(
            data=frame, x="Epoch", y="Cross Entropy" if ce else "Accuracy",
            hue="Case", col="Convolve", style="Case", kind="line",
        )
    )
    figure = grids.figure
    figure.savefig(
        os.path.join(
            "figure", "cnn_amprec{:s}.png".format("ce" if ce else "acc"),
        ),
    )
    plt.close(figure)

def task_cnn(ce: bool, /) -> None:
    R"""
    CNN task.
    """
    #
    buf_runtime = []
    buf_frame = []
    (runtime, frame) = (
                load(
                    seed=47, shuffle=False, batch_size=100, cnn=False,
                    cgcnn=False, kernel=5, stride=1, amprec=False,
                    optim_alg="default", lr=0.001, wd=0.0, ce=ce, file="daaea1279601ff5bf44d98bada124805"
                )
            )
    buf_runtime.append(runtime)
    buf_frame.append(frame)
    frame = pd.concat(buf_frame, ignore_index=True)

    #
    grids = (
        sns.relplot(
            data=frame, x="Epoch", y="Cross Entropy" if ce else "Accuracy",
            hue="Case", col="Convolve", style="Case", kind="line",
        )
    )
    figure = grids.figure
    figure.savefig(
        os.path.join(
            "figure", "cnn_{:s}.png".format("ce" if ce else "acc"),
        ),
    )
    plt.close(figure)


def main(*ARGS):
    R"""
    Main.
    """
    # YOU SHOULD FILL IN THIS FUNCTION
    ...

    #
    parser = argparse.ArgumentParser(description="Visualization Execution")
    parser.add_argument(
        "--ce",
        action="store_true", help="Visualize training cross entropy loss.",
    )
    parser.add_argument(
        "--minibatch",
        action="store_true", help="Visualize minibatch task.",
    )
    parser.add_argument(
        "--optimizer",
        action="store_true", help="Visualize optimizer task.",
    )
    parser.add_argument(
        "--regularization",
        action="store_true", help="Visualize regularization task.",
    )
    parser.add_argument(
        "--amp",
        action="store_true", help="Visualize CNN with amprec task.",
    )
    parser.add_argument(
        "--cnn",
        action="store_true", help="Visualize CNN task.",
    )
    args = parser.parse_args() if len(ARGS) == 0 else parser.parse_args(ARGS)

    # Parse the command line arguments.
    ce = args.ce
    cnn = args.cnn
    amp= args.amp

    #
    if not os.path.isdir("figure"):
        #
        os.makedirs("figure")

    if cnn:
        #
        task_cnn(ce)

    if amp:
        #
        task_amprec(ce)


#
if __name__ == "__main__":
    #
    main()