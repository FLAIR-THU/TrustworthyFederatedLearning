import argparse
import glob
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def add_args(parser):
    parser.add_argument(
        "-p",
        "--path_to_dir",
        type=str,
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parsed_args = add_args(parser)

    round_num = max(
        [
            int(p.split("_")[-1].split(".")[0])
            for p in glob.glob(os.path.join(parsed_args.path_to_dir, "temp_lp_tree_*"))
        ]
    )

    leaf_purity_mean = np.zeros(round_num)
    leaf_purity_std = np.zeros(round_num)
    loss_mean = np.zeros(round_num)
    loss_std = np.zeros(round_num)
    for i in range(1, round_num + 1):
        with open(
            os.path.join(parsed_args.path_to_dir, f"temp_lp_tree_{i}.out"), mode="r"
        ) as f:
            leaf_purity = [float(s.strip()) for s in f.readlines()]
            leaf_purity_mean[i - 1] = np.mean(leaf_purity)
            leaf_purity_std[i - 1] = np.std(leaf_purity)
        with open(
            os.path.join(parsed_args.path_to_dir, f"temp_loss_tree_{i}.out"), mode="r"
        ) as f:
            loss = [float(s.strip()) for s in f.readlines()]
            loss_mean[i - 1] = np.mean(loss)
            loss_std[i - 1] = np.std(loss)

    df = pd.DataFrame(
        columns=["loss_mean", "loss_std" "leaf_purity_mean", "leaf_purity_std"]
    )
    df["loss_mean"] = loss_mean
    df["loss_std"] = loss_std
    df["leaf_purity_mean"] = leaf_purity_mean
    df["leaf_purity_std"] = leaf_purity_std
    df.to_csv(os.path.join(parsed_args.path_to_dir, "loss_lp.csv"))

    plt.style.use("ggplot")
    fig = plt.figure()
    fig.add_subplot(1, 2, 1)
    plt.plot(list(range(1, round_num + 1)), leaf_purity_mean, color="#1B2ACC")
    plt.fill_between(
        list(range(1, round_num + 1)),
        leaf_purity_mean - leaf_purity_std,
        leaf_purity_mean + leaf_purity_std,
        alpha=0.2,
        edgecolor="#1B2ACC",
        facecolor="#089FFF",
    )
    plt.xlabel("round")
    plt.title("leaf purity")
    fig.add_subplot(1, 2, 2)
    plt.plot(list(range(1, round_num + 1)), loss_mean, color="#CC4F1B")
    plt.fill_between(
        list(range(1, round_num + 1)),
        loss_mean - loss_std,
        loss_mean + loss_std,
        alpha=0.5,
        edgecolor="#CC4F1B",
        facecolor="#FF9848",
    )
    plt.xlabel("round")
    plt.title("loss")

    plt.tight_layout()
    plt.savefig(os.path.join(parsed_args.path_to_dir, "result.png"))

    with open(
        os.path.join(parsed_args.path_to_dir, "temp_train_auc.out"), mode="r"
    ) as f:
        train_auc = [float(s.strip()) for s in f.readlines()]
    with open(os.path.join(parsed_args.path_to_dir, "temp_val_auc.out"), mode="r") as f:
        val_auc = [float(s.strip()) for s in f.readlines()]

    print(
        f"AUC (train): {np.round(np.mean(train_auc), decimals=4)}±{np.round(np.std(train_auc), decimals=4)}"
    )
    print(
        f"AUC (validation): {np.round(np.mean(val_auc), decimals=4)}±{np.round(np.std(val_auc), decimals=4)}"
    )

    leak_csv = pd.concat(
        [
            pd.read_csv(p)
            for p in glob.glob(os.path.join(parsed_args.path_to_dir, "*_leak.csv"))
        ]
    )
    leak_csv.to_csv(os.path.join(parsed_args.path_to_dir, "leak.csv"))

    try:
        for score_type in ["c", "h", "v", "p", "ip", "f"]:
            baseline_mean = np.round(
                leak_csv[f"baseline_{score_type}"].mean(), decimals=4
            )
            baseline_std = np.round(
                leak_csv[f"baseline_{score_type}"].std(), decimals=4
            )
            our_mean = np.round(leak_csv[f"our_{score_type}"].mean(), decimals=4)
            our_std = np.round(leak_csv[f"our_{score_type}"].std(), decimals=4)
            print(f"{score_type} (baseline): {baseline_mean}±{baseline_std}")
            print(f"{score_type} (our): {our_mean}±{our_std}")
    except:
        for score_type in ["a"]:
            baseline_mean = np.round(
                leak_csv[f"baseline_{score_type}"].mean(), decimals=4
            )
            baseline_std = np.round(
                leak_csv[f"baseline_{score_type}"].std(), decimals=4
            )
            our_mean = np.round(leak_csv[f"our_{score_type}"].mean(), decimals=4)
            our_std = np.round(leak_csv[f"our_{score_type}"].std(), decimals=4)
            print(f"{score_type} (baseline): {baseline_mean}±{baseline_std}")
            print(f"{score_type} (our): {our_mean}±{our_std}")
