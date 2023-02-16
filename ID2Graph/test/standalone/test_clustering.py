import numpy as np


def test_fmeasure():
    from llatvfl.clustering import get_f_p_r

    f, _, _ = get_f_p_r(
        [0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 2, 1, 0, 2, 2, 2, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
    )

    np.testing.assert_array_almost_equal(f, 0.7069009421, decimal=6)
