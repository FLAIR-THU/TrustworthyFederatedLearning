import argparse

import numpy as np
from sklearn import metrics, preprocessing
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

from llatvfl.clustering import get_f_p_r

# from matplotlib import pyplot as plt

label2maker = {0: "o", 1: "x"}


def add_args(parser):
    parser.add_argument(
        "-p",
        "--path_to_input_file",
        type=str,
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
    )
    parser.add_argument(
        "-q",
        "--path_to_union_file",
        type=str,
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parsed_args = add_args(parser)

    clustering_cls = KMeans

    print(
        "baseline_c,baseline_h,baseline_v,baseline_p,baseline_ip,baseline_f,our_c,our_h,our_v,our_p,our_ip,our_f"
    )

    with open(parsed_args.path_to_input_file, mode="r") as f:
        lines = f.readlines()
        first_line = lines[0].split(" ")
        num_classes, num_row, num_col, num_party = (
            int(first_line[0]),
            int(first_line[1]),
            int(first_line[2]),
            int(first_line[3]),
        )

        start_line_num_of_active_party = 3 + int(lines[1][:-1])
        X_train = np.array(
            [
                lines[col_idx][:-1].split(" ")
                for col_idx in range(
                    start_line_num_of_active_party,
                    start_line_num_of_active_party
                    + int(lines[start_line_num_of_active_party - 1][:-1]),
                )
            ]
        )
        min_max_scaler = preprocessing.MinMaxScaler()
        X_train_minmax = min_max_scaler.fit_transform(X_train.T)

        y_train = lines[num_col + num_party + 1].split(" ")
        y_train = np.array([int(y) for y in y_train])
        unique_labels = np.unique(y_train)

    with open(parsed_args.path_to_union_file, mode="r") as f:
        lines = f.readlines()
        union_clusters = lines[0].split(" ")[:-1]

    union_clusters = LabelEncoder().fit_transform(union_clusters)

    c_score_baseline = metrics.completeness_score(y_train, union_clusters)
    h_score_baseline = metrics.homogeneity_score(y_train, union_clusters)
    v_score_baseline = metrics.v_measure_score(y_train, union_clusters)

    f_score_baseline, p_score_baseline, ip_score_baseline = get_f_p_r(
        y_train, union_clusters
    )

    num_union = len(np.unique(union_clusters))
    X_com = np.zeros((num_row, num_union))
    for i in range(num_row):
        X_com[i, union_clusters[i]] = 1

    kmeans_with_com = clustering_cls(
        n_clusters=num_classes, random_state=parsed_args.seed
    ).fit(np.hstack([X_train_minmax, X_com]))
    c_score_with_com = metrics.completeness_score(y_train, kmeans_with_com.labels_)
    h_score_with_com = metrics.homogeneity_score(y_train, kmeans_with_com.labels_)
    v_score_with_com = metrics.v_measure_score(y_train, kmeans_with_com.labels_)

    f_score_with_com, p_score_with_com, ip_score_with_com = get_f_p_r(
        y_train, kmeans_with_com.labels_
    )

    print(
        f"{c_score_baseline},{h_score_baseline},{v_score_baseline},{p_score_baseline},{ip_score_baseline},{f_score_baseline},{c_score_with_com},{h_score_with_com},{v_score_with_com},{p_score_with_com},{ip_score_with_com},{f_score_with_com}"
    )
