import numpy as np

def generate_res_func(y_pred,y_true):
    assert y_pred.shape == y_true.shape
    return np.abs(y_pred-y_true)