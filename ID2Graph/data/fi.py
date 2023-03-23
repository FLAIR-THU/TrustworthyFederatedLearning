import argparse
import os
import random

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.preprocessing import LabelEncoder


def add_args(parser):
    parser.add_argument(
        "-d",
        "--dataset_type",
        type=str,
    )

    parser.add_argument(
        "-p",
        "--path_to_dir",
        type=str,
    )

    parser.add_argument(
        "-n",
        "--num_samples",
        type=int,
        default=-1,
    )

    parser.add_argument(
        "-f",
        "--feature_num_ratio_of_active_party",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "-v",
        "--feature_num_ratio_of_passive_party",
        type=float,
        default=-1,
    )

    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()
    return args


def sampling_col_alloc(
    col_num, feature_num_ratio_of_active_party, feature_num_ratio_of_passive_party
):
    shufled_col_indicies = random.sample(list(range(col_num)), col_num)
    col_num_of_active_party = max(1, int(feature_num_ratio_of_active_party * col_num))
    if feature_num_ratio_of_passive_party < 0:
        col_alloc = [
            shufled_col_indicies[:col_num_of_active_party],
            shufled_col_indicies[col_num_of_active_party:],
        ]
    else:
        col_num_of_passive_party = max(
            1, int(feature_num_ratio_of_passive_party * col_num)
        )
        col_alloc = [
            shufled_col_indicies[:col_num_of_active_party],
            shufled_col_indicies[
                col_num_of_active_party : (
                    min(
                        col_num_of_active_party + col_num_of_passive_party,
                        col_num,
                    )
                )
            ],
        ]

    return col_alloc


def convert_df_to_input(
    X_train,
    y_train,
    X_val,
    y_val,
    output_path,
    col_alloc=None,
    parties_num=2,
    feature_num_ratio_of_active_party=0.5,
    feature_num_ratio_of_passive_party=-1,
    replace_nan="-1",
):
    row_num_train, col_num = X_train.shape
    row_num_val = X_val.shape[0]

    if col_alloc is None:
        col_alloc = sampling_col_alloc(
            col_num,
            feature_num_ratio_of_active_party,
            feature_num_ratio_of_passive_party,
        )

    with open(output_path, mode="w") as f:
        f.write(
            f"{len(list(set(y_train)))} {row_num_train} {len(sum(col_alloc, []))} {parties_num}\n"
        )
        for ca in col_alloc:
            f.write(f"{len(ca)}\n")
            for i in ca:
                f.write(
                    " ".join(
                        [
                            str(x) if not np.isnan(x) else replace_nan
                            for x in X_train[:, i]
                        ]
                    )
                    + "\n"
                )
        f.write(" ".join([str(y) for y in y_train]) + "\n")
        f.write(f"{row_num_val}\n")
        for ca in col_alloc:
            for i in ca:
                f.write(
                    " ".join(
                        [
                            str(x) if not np.isnan(x) else replace_nan
                            for x in X_val[:, i]
                        ]
                    )
                    + "\n"
                )
        f.write(" ".join([str(y) for y in y_val]))


def sampling(df, yname, parsed_args):
    if parsed_args.num_samples == -1:
        return df
    else:
        pos_df = df[df[yname] == 1]
        neg_df = df[df[yname] == 0]
        pos_num = int(parsed_args.num_samples / 2)
        neg_num = parsed_args.num_samples - pos_num
        pos_df = pos_df.sample(pos_num)
        neg_df = neg_df.sample(neg_num)
        df_ = pd.concat([pos_df, neg_df])
        return df_


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parsed_args = add_args(parser)

    random.seed(parsed_args.seed)
    np.random.seed(parsed_args.seed)
    col_alloc = None

    if parsed_args.dataset_type == "avila":
        df_tr = pd.read_csv(
            os.path.join(parsed_args.path_to_dir, "avila-tr.txt"),
            header=None,
        )
        df_ts = pd.read_csv(
            os.path.join(parsed_args.path_to_dir, "avila-ts.txt"),
            header=None,
        )
        df = pd.concat([df_tr, df_ts], axis=0)
        string2int = {
            s: i
            for i, s in enumerate(
                ["A", "B", "C", "D", "E", "F", "G", "H", "I", "W", "X", "Y"]
            )
        }
        df[10] = df[10].apply(lambda x: string2int[x])
        df = sampling(df, 10, parsed_args)

        X = df[list(range(10))].values
        y = df[10].values

    elif parsed_args.dataset_type == "phishing":
        df = pd.read_csv(
            os.path.join(parsed_args.path_to_dir, "phishing.data"), header=None
        )
        df[30] = df[30].apply(lambda x: 0 if x == -1 else 1)
        df = sampling(df, 30, parsed_args)

        X = df[list(range(30))].values
        y = df[30].values

    elif parsed_args.dataset_type == "drive":
        df = pd.read_csv(
            os.path.join(parsed_args.path_to_dir, "Sensorless_drive_diagnosis.txt"),
            sep=" ",
            header=None,
        )

        X = df[list(range(48))].values
        y = df[48].values - 1

    elif parsed_args.dataset_type == "nursery":
        df = pd.read_csv(
            os.path.join(parsed_args.path_to_dir, "nursery.data"), header=None
        )
        df[8] = LabelEncoder().fit_transform(df[8].values)

        col_alloc_origin = sampling_col_alloc(
            col_num=df.shape[1] - 1,
            feature_num_ratio_of_active_party=parsed_args.feature_num_ratio_of_active_party,
            feature_num_ratio_of_passive_party=parsed_args.feature_num_ratio_of_passive_party,
        )
        X_d = df.drop(8, axis=1)
        X_a = pd.get_dummies(X_d[X_d.columns[col_alloc_origin[0]]], drop_first=True)
        X_p = pd.get_dummies(X_d[X_d.columns[col_alloc_origin[1]]], drop_first=True)
        col_alloc = [
            list(range(X_a.shape[1])),
            list(range(X_a.shape[1], X_a.shape[1] + X_p.shape[1])),
        ]
        X = pd.concat([X_a, X_p], axis=1).values
        y = df[8].values

    elif parsed_args.dataset_type == "fraud":
        df = pd.read_csv(
            os.path.join(parsed_args.path_to_dir, "fraud_detection_bank_dataset.csv")
        )

        X = df[[f"col_{i}" for i in range(112)]].values
        y = df["targets"].values

    elif parsed_args.dataset_type == "ucicreditcard":
        df = pd.read_csv(os.path.join(parsed_args.path_to_dir, "UCI_Credit_Card.csv"))
        df = sampling(df, "default.payment.next.month", parsed_args)

        X = df[
            [
                "LIMIT_BAL",
                "SEX",
                "EDUCATION",
                "MARRIAGE",
                "AGE",
                "PAY_0",
                "PAY_2",
                "PAY_3",
                "PAY_4",
                "PAY_5",
                "PAY_6",
                "BILL_AMT1",
                "BILL_AMT2",
                "BILL_AMT3",
                "BILL_AMT4",
                "BILL_AMT5",
                "BILL_AMT6",
                "PAY_AMT1",
                "PAY_AMT2",
                "PAY_AMT3",
                "PAY_AMT4",
                "PAY_AMT5",
                "PAY_AMT6",
            ]
        ].values
        y = df["default.payment.next.month"].values

    else:
        raise ValueError(f"{parsed_args.dataset_type} is not supported.")

    mm = preprocessing.MinMaxScaler()
    X_minmax = mm.fit_transform(X)
    selector = SelectKBest(mutual_info_classif, k=X.shape[1] * 0.5)
    selector.fit(X_minmax, y)

    np.save(
        os.path.join(
            parsed_args.path_to_dir,
            f"{parsed_args.dataset_type}_fti",
        ),
        selector.scores_,
    )
