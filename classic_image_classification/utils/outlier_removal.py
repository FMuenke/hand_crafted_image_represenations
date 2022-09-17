import numpy as np


# get_curve_online() is defined in: https://github.com/iCGY96/ARPL/blob/master/core/evaluation.py
def get_curve_online(known, novel, stypes = ['Bas'], tpr_best=None, fpr_best=None):
    tp, fp = dict(), dict()
    tnr_at_tpr95 = dict()
    best_threshold = None
    for stype in stypes:
        known.sort()
        novel.sort()
        end = np.max([np.max(known), np.max(novel)])
        start = np.min([np.min(known),np.min(novel)])
        num_k = known.shape[0]
        num_n = novel.shape[0]
        tp[stype] = -np.ones([num_k+num_n+1], dtype=int)
        fp[stype] = -np.ones([num_k+num_n+1], dtype=int)
        tp[stype][0], fp[stype][0] = num_k, num_n
        k, n = 0, 0
        for l in range(num_k+num_n):
            if k == num_k:
                tp[stype][l+1:] = tp[stype][l]
                fp[stype][l+1:] = np.arange(fp[stype][l]-1, -1, -1)
                break
            elif n == num_n:
                tp[stype][l+1:] = np.arange(tp[stype][l]-1, -1, -1)
                fp[stype][l+1:] = fp[stype][l]
                break
            else:
                if novel[n] < known[k]:
                    n += 1
                    tp[stype][l+1] = tp[stype][l]
                    fp[stype][l+1] = fp[stype][l] - 1
                else:
                    k += 1
                    tp[stype][l+1] = tp[stype][l] - 1
                    fp[stype][l+1] = fp[stype][l]

            if tpr_best and fpr_best:
              if tp[stype][l+1]/tp[stype][0] == tpr_best and fp[stype][l+1]/fp[stype][0] == fpr_best:
                  best_threshold = novel[n]
        tpr95_pos = np.abs(tp[stype] / num_k - .95).argmin()
        tnr_at_tpr95[stype] = 1. - fp[stype][tpr95_pos] / num_n
    return tp, fp, tnr_at_tpr95, best_threshold


def get_best_threshold(score, status):

    x1 = score[status == 1]
    x2 = score[status == 0]

    # get TP, FP, TNR@TPR95; TPR and FPR
    tp, fp, tnr_at_tpr95, _ = get_curve_online(x1, x2, ['Bas'])
    tpr = np.concatenate([[1.], tp['Bas'] / tp['Bas'][0], [0.]])
    fpr = np.concatenate([[1.], fp['Bas'] / fp['Bas'][0], [0.]])

    # get best threshold
    J = tpr - fpr
    ix = np.argmax(J)
    _, _, _, best_threshold = get_curve_online(x1, x2, ['Bas'], tpr_best=tpr[ix], fpr_best=fpr[ix])
    return best_threshold
