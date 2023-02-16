import warnings

import numpy as np
from sklearn import metrics


def get_f_p_r(y, y_hat):
    cm_matrix = metrics.cluster.contingency_matrix(y, y_hat)

    num_label = len(np.unique(y))
    num_cluster = len(np.unique(y_hat))

    warnings.resetwarnings()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p_scores = cm_matrix / np.repeat(
            cm_matrix.sum(axis=0).reshape(1, -1), num_label, axis=0
        )
        r_scores = cm_matrix / np.repeat(
            cm_matrix.sum(axis=1).reshape(-1, 1), num_cluster, axis=1
        )
        f_scores = 2 * r_scores * p_scores / (r_scores + p_scores)

    p_score = np.sum(
        np.nanmax(p_scores, axis=0) * cm_matrix.sum(axis=0) / cm_matrix.sum()
    )
    r_score = np.sum(
        np.nanmax(r_scores, axis=1) * cm_matrix.sum(axis=1) / cm_matrix.sum()
    )
    f_score = np.sum(
        np.nanmax(f_scores, axis=1) * cm_matrix.sum(axis=1) / cm_matrix.sum()
    )
    return f_score, p_score, r_score
