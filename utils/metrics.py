import numpy as np

def compute_metrics(predicts, labels):
    N,  H, W = predicts.shape

    predicts = predicts.reshape((-1, H*W))
    labels = labels.reshape((-1, H*W))

    sum_p = np.sum(predicts, axis=1)
    sum_l = np.sum(labels, axis=1)
    intersection = np.sum(np.logical_and(predicts, labels), axis=1)

    numer = 2*intersection
    denom = sum_p + sum_l
    dice = numer / (denom + 1e-6)

    empty_indices = np.where(sum_l <= 0)[0]
    non_empty_indices = np.where(sum_l > 0)[0]
    if len(non_empty_indices) == 0:
        non_empty_mean_dice = 0.0
    else:
        non_empty_dice = dice[non_empty_indices]
        non_empty_mean_dice = float(np.mean(non_empty_dice))

    all_non_empty_index = np.where(numer > 0)[0]
    all_empty_index = np.where(denom == 0)[0]
    dice[all_empty_index] = 1
    mean_dice = float(np.mean(dice))

    cls_accuracy = (len(all_non_empty_index) + len(all_empty_index)) / N

    correct_indices = np.where((sum_p > 0) == (sum_l > 0))[0]
    incorrect_indices = np.where((sum_p > 0) != (sum_l > 0))[0]

    tp = len(np.where(sum_l[correct_indices] > 0)[0])
    tn = len(np.where(sum_l[correct_indices] == 0)[0])

    fp = len(np.where(sum_l[incorrect_indices] == 0)[0])
    fn = len(np.where(sum_l[incorrect_indices] > 0)[0])

    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)

    tnr = tn / (tn + fp + 1e-10)
    fpr = fp / (fp + tn + 1e-10)

    return mean_dice, non_empty_mean_dice, cls_accuracy, precision, recall, tnr, fpr