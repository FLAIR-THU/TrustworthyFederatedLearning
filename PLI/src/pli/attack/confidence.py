import math


def get_pj(n, alpha):
    if alpha == 0:
        return 1
    else:
        return math.exp(1 / alpha) / (n - 1 + math.exp(1 / alpha))


def get_pi(n, alpha):
    if alpha == 0:
        return 0
    else:
        return 1 / (n - 1 + math.exp(1 / alpha))


def get_alpha(n, pj):
    if pj == 1:
        return 0
    else:
        return 1 / math.log(pj * (n - 1) / (1 - pj))
