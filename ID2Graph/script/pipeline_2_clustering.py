import argparse

import numpy as np
from sklearn import metrics, preprocessing
from sklearn.cluster import KMeans

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
        "-q",
        "--path_to_com_file",
        type=str,
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
    )
    parser.add_argument(
        "-k",
        "--weight_for_community_variables",
        type=float,
        default=1.0,
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

    kmeans = clustering_cls(n_clusters=num_classes, random_state=parsed_args.seed).fit(
        X_train_minmax
    )
    c_score_baseline = metrics.completeness_score(y_train, kmeans.labels_)
    h_score_baseline = metrics.homogeneity_score(y_train, kmeans.labels_)
    v_score_baseline = metrics.v_measure_score(y_train, kmeans.labels_)

    f_score_baseline, p_score_baseline, ip_score_baseline = get_f_p_r(
        y_train, kmeans.labels_
    )
    cm_matrix = metrics.cluster.contingency_matrix(y_train, kmeans.labels_)

    with open(parsed_args.path_to_com_file, mode="r") as f:
        lines = f.readlines()
        comm_num = int(lines[0])
        node_num = int(lines[1])
        X_com = np.zeros((num_row, comm_num))

        for i in range(comm_num):
            temp_nodes_in_comm = lines[i + 2].split(" ")[:-1]
            for k in temp_nodes_in_comm:
                X_com[int(k), i] += parsed_args.weight_for_community_variables

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
