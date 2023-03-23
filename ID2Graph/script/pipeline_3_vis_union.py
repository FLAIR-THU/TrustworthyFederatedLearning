import argparse
import glob
import os

import networkx as nx
import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

# from matplotlib import pyplot as plt

label2maker = {0: "o", 1: "x"}


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

    clustering_cls = KMeans

    parser = argparse.ArgumentParser()
    parsed_args = add_args(parser)
    list_union_files = glob.glob(os.path.join(parsed_args.path_to_dir, "*_union.out"))

    for path_to_union_file in list_union_files:
        path_to_input_file = path_to_union_file.split("_")[0] + "_data.in"
        with open(path_to_input_file, mode="r") as f:
            lines = f.readlines()
            first_line = lines[0].split(" ")
            num_classes, num_row, num_col, num_party = (
                int(first_line[0]),
                int(first_line[1]),
                int(first_line[2]),
                int(first_line[3]),
            )

            y_train = lines[num_col + num_party + 1].split(" ")
            y_train = [int(y) for y in y_train]

        with open(path_to_union_file, mode="r") as f:
            lines = f.readlines()
            union_clusters = lines[0].split(" ")[:-1]

        edge_color = []
        edge_list = []
        adj_mat = np.zeros((len(union_clusters), len(union_clusters)))
        for i, c in enumerate(union_clusters):
            if i != int(c):
                adj_mat[i, int(c)] += 1
                adj_mat[int(c), i] += 1
                edge_list.append((i, int(c)))
                edge_color.append(0)

        union_clusters = LabelEncoder().fit_transform(union_clusters)

        plt.style.use("ggplot")
        cmap = cm.get_cmap("plasma", len(set(union_clusters)) + 3)
        G = nx.from_numpy_matrix(
            adj_mat, create_using=nx.MultiGraph, parallel_edges=False
        )
        pos = nx.spring_layout(G)

        label2color = {0: "g", 1: "r", 2: "b", 3: "k", 4: "y", 5: "m"}
        nx.draw_networkx(
            G,
            pos,
            with_labels=False,
            alpha=0.7,
            node_size=10,
            width=0.2,
            cmap=cmap,
            edge_color=edge_color,
            edgelist=edge_list,
            node_color=[label2color[y] for y in y_train],
        )

        plt.savefig(
            path_to_union_file.split(".")[0] + "_plot.png",
            bbox_inches="tight",
            pad_inches=0,
            format="png",
            dpi=300,
        )
        plt.close()
