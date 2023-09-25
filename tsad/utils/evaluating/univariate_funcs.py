import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from .src import extract_cp_confusion_matrix, filter_detecting_boundaries


def confusion_matrix(true, prediction):
    true_ = true == 1
    prediction_ = prediction == 1
    TP = (true_ & prediction_).sum()
    TN = (~true_ & ~prediction_).sum()
    FP = (~true_ & prediction_).sum()
    FN = (true_ & ~prediction_).sum()
    return TP, TN, FP, FN


def single_average_delay(detecting_boundaries, prediction, anomaly_window_destination, clear_anomalies_mode):
    """
    anomaly_window_destination: 'lefter', 'righter', 'center'. Default='right'
    """
    detecting_boundaries = filter_detecting_boundaries(detecting_boundaries)
    point = 0 if clear_anomalies_mode else -1
    dict_cp_confusion = extract_cp_confusion_matrix(detecting_boundaries, prediction, point=point)

    missing = 0
    detectHistory = []
    all_true_anom = 0
    FP = 0

    FP += len(dict_cp_confusion['FPs'])
    missing += len(dict_cp_confusion['FNs'])
    all_true_anom += len(dict_cp_confusion['TPs']) + len(dict_cp_confusion['FNs'])

    if anomaly_window_destination == 'lefter':
        def average_time(output_cp_cm_tp):
            return output_cp_cm_tp[2] - output_cp_cm_tp[1]
    elif anomaly_window_destination == 'righter':
        def average_time(output_cp_cm_tp):
            return output_cp_cm_tp[1] - output_cp_cm_tp[0]
    elif anomaly_window_destination == 'center':
        def average_time(output_cp_cm_tp):
            return output_cp_cm_tp[1] - (output_cp_cm_tp[0] + (output_cp_cm_tp[2] - output_cp_cm_tp[0]) / 2)
    else:
        raise Exception("Choose anomaly_window_destination")

    for fp_case_window in dict_cp_confusion['TPs']:
        detectHistory.append(average_time(dict_cp_confusion['TPs'][fp_case_window]))
    return missing, detectHistory, FP, all_true_anom


def my_scale(fp_case_window=None,
             A_tp=1,
             A_fp=0,
             koef=1,
             detalization=1000,
             clear_anomalies_mode=True,
             plot_figure=False):
    """
    ts - segment on which the window is applied
    """
    x = np.linspace(-np.pi / 2, np.pi / 2, detalization)
    x = x if clear_anomalies_mode else x[::-1]
    y = (A_tp - A_fp) / 2 * -1 * np.tanh(koef * x) / (np.tanh(np.pi * koef / 2)) + (A_tp - A_fp) / 2 + A_fp
    if not plot_figure:
        event = int((fp_case_window[1] - fp_case_window[0]) / \
                    (fp_case_window[-1] - fp_case_window[0]) * detalization)
        if event >= len(x):
            event = len(x) - 1
        score = y[event]
        return score
    else:
        return y


def single_evaluate_nab(detecting_boundaries,
                        prediction,
                        table_of_coef=None,
                        clear_anomalies_mode=True,
                        scale_func="improved",
                        scale_koef=1,
                        plot_figure=True  # TODO
                        ):
    """
    
    detecting_boundaries: list of list of two float values
                The list of lists of left and right boundary indices
                for scoring results of labeling if empty. Can be [[]], or [[],[t1,t2],[]]                
    table_of_coef: pandas array (3x4) of float values
                Table of coefficients for NAB score function
                indices: 'Standard','LowFP','LowFN'
                columns:'A_tp','A_fp','A_tn','A_fn'
                
    scale_func {default}, improved
    недостатки scale_func default  -
    1 - зависит от относительного шага, а это значит, что если 
    слишком много точек в scoring window то перепад будет слишком
    жестким в середение. 
    2-   то самая левая точка не равно  Atp, а права не равна Afp
    (особенно если пррименять расплывающую множитель)

    clear_anomalies_mode тогда слева от границы Atp срправа Afp,
    иначе fault mode, когда слева от границы Afp срправа Atp
    """

    #     def sigm_scale(len_ts, A_tp, A_fp, koef=1):
    #         x = np.arange(-int(len_ts/2), len_ts - int(len_ts/2))

    #         x = x if clear_anomalies_mode else x[::-1]
    #         y = (A_tp-A_fp)*(1/(1+np.exp(5*x*koef))) + A_fp
    #         return y
    #     def my_scale(len_ts,A_tp,A_fp,koef=1):
    #         """ts - участок на котором надо жахнуть окно """
    #         x = np.linspace(-np.pi/2,np.pi/2,len_ts)
    #         x = x if clear_anomalies_mode else x[::-1]
    #         # Приведение если неравномерный шаг.
    #         #x_new = x_old * ( np.pi / (x_old[-1]-x_old[0])) - x_old[0]*( np.pi / (x_old[-1]-x_old[0])) - np.pi/2
    #         y = (A_tp-A_fp)/2*-1*np.tanh(koef*x)/(np.tanh(np.pi*koef/2)) + (A_tp-A_fp)/2 + A_fp
    #         return y

    if scale_func == "improved":
        scale_func = my_scale
    #     elif scale_func == "default":
    #         scale_func = sigm_scale
    else:
        raise Exception("choose the scale_func")

    # filter
    detecting_boundaries = filter_detecting_boundaries(detecting_boundaries)

    if table_of_coef is None:
        table_of_coef = pd.DataFrame([[1.0, -0.11, 1.0, -1.0],
                                      [1.0, -0.22, 1.0, -1.0],
                                      [1.0, -0.11, 1.0, -2.0]])
        table_of_coef.index = ['Standard', 'LowFP', 'LowFN']
        table_of_coef.index.name = "Metric"
        table_of_coef.columns = ['A_tp', 'A_fp', 'A_tn', 'A_fn']

    # GO
    point = 0 if clear_anomalies_mode else -1
    dict_cp_confusion = extract_cp_confusion_matrix(detecting_boundaries, prediction, point=point)

    Scores, Scores_perfect, Scores_null = [], [], []
    for profile in ['Standard', 'LowFP', 'LowFN']:
        A_tp = table_of_coef['A_tp'][profile]
        A_fp = table_of_coef['A_fp'][profile]
        A_fn = table_of_coef['A_fn'][profile]

        score = 0
        score += A_fp * len(dict_cp_confusion['FPs'])
        score += A_fn * len(dict_cp_confusion['FNs'])
        for fp_case_window in dict_cp_confusion['TPs']:
            set_times = dict_cp_confusion['TPs'][fp_case_window]
            score += scale_func(set_times, A_tp, A_fp, koef=scale_koef)

        Scores.append(score)
        Scores_perfect.append(len(detecting_boundaries) * A_tp)
        Scores_null.append(len(detecting_boundaries) * A_fn)

    return np.array([np.array(Scores), np.array(Scores_null), np.array(Scores_perfect)])
