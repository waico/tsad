# input y_pred,y_true of 3d, 

import numpy as np

def abs(y_pred,y_true):
    """
    Функция позволяющая получить разницу
    """
    assert y_pred.shape == y_true.shape
    return np.abs(y_pred-y_true)