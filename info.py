from math import log
import numpy as np


def mutual_info(p_x, p_y, p_xy, eps=1e-06):
    h_x = p_entropy(p_x, eps)
    h_y = p_entropy(p_y, eps)
    h_xy = p_entropy(p_xy, eps)
    return h_x + h_y - h_xy


def p_entropy(p_x, eps=1e-06):
    if not isinstance(p_x, np.ndarray):
        p_x = np.asarray(p_x)
    if np.any(p_x == 0):
        p_x = p_x + eps
    p_sum = np.sum(p_x)
    return -np.sum((p_x / p_sum) * (np.log(p_x) - log(p_sum)))
