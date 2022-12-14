import numpy as np
from scipy import stats
from sklearn import metrics
import torch
from sklearn.metrics import precision_recall_curve, auc, confusion_matrix, f1_score, classification_report, roc_auc_score, average_precision_score


def d_prime(auc):
    standard_normal = stats.norm()
    d_prime = standard_normal.ppf(auc) * np.sqrt(2.0)
    return d_prime

def calculate_stats(output, target):
    """Calculate statistics including mAP, AUC, etc.

    Args:
      output: 2d array, (samples_num, classes_num)
      target: 2d array, (samples_num, classes_num)

    Returns:
      stats: list of statistic of each class.
    """

    classes_num = target.shape[-1]
    stats = []

    # Accuracy, only used for single-label classification such as esc-50, not for multiple label one such as AudioSet
    acc = metrics.accuracy_score(np.argmax(target, 1), np.argmax(output, 1))

    # Class-wise statistics
    for k in range(classes_num):

        # Average precision
        avg_precision = metrics.average_precision_score(
            target[:, k], output[:, k], average=None)

        # AUC
        auc = metrics.roc_auc_score(target[:, k], output[:, k], average=None)

        # Precisions, recalls
        (precisions, recalls, thresholds) = metrics.precision_recall_curve(
            target[:, k], output[:, k])

        # FPR, TPR
        (fpr, tpr, thresholds) = metrics.roc_curve(target[:, k], output[:, k])

        save_every_steps = 1000     # Sample statistics to reduce size
        dict = {'precisions': precisions[0::save_every_steps],
                'recalls': recalls[0::save_every_steps],
                'AP': avg_precision,
                'fpr': fpr[0::save_every_steps],
                'fnr': 1. - tpr[0::save_every_steps],
                'auc': auc,
                # note acc is not class-wise, this is just to keep consistent with other metrics
                'acc': acc
                }
        stats.append(dict)

    return stats

def metric_fn_openmic(Y_true, Y_mask, preds, threshold=0.5):

    # Get classwise classification metrics
    avg_fscore_weighted = []
    avg_fscore_macro = []
    avg_precision_macro = []
    avg_recall_macro = []
    precision_0 = []
    recall_0 = []
    precision_1 = []
    recall_1 = []
    auroc = []
    pr_auc = []
    
    #print(preds)
    
    
    for i in range(Y_true.shape[-1]):
        labels = Y_true[:, i]
        labels_mask = Y_mask[:, i]
        predictions = preds[:, i]

        # Get relevant indices from the mask
        relevant_inds = np.where(labels_mask)[0]
        
        # get AUC before thresholding
        auc = roc_auc_score(labels[relevant_inds], predictions[relevant_inds])
        ap = average_precision_score(labels[relevant_inds], predictions[relevant_inds])
        # Binarize the predictions based on the threshold.
        predictions[predictions >= threshold] = 1
        predictions[predictions < 1] = 0
        #labels[labels >= threshold] = 1
        #labels[labels < 1] = 0
        
        #print(labels)
        #print(predictions)
#     print(classification_report(
#         labels[relevant_inds], predictions[relevant_inds]))
    # return classification report
        results = classification_report(labels[relevant_inds], predictions[relevant_inds], output_dict=True)
        #print(results)

        avg_fscore_weighted.append(results['weighted avg']['f1-score'])
        avg_fscore_macro.append(results['macro avg']['f1-score'])
        avg_precision_macro.append(results['macro avg']['precision'])
        avg_recall_macro.append(results['macro avg']['recall'])
        precision_0.append(results['0.0']['precision'])
        recall_0.append(results['0.0']['recall'])
        precision_1.append(results['1.0']['precision'])
        recall_1.append(results['1.0']['recall'])
        auroc.append(auc)
        pr_auc.append(ap)
        

    metrics = {
        'F1_macro': np.array(avg_fscore_macro),
        'F1_weighted': np.array(avg_fscore_weighted),
        'precision': np.array(avg_precision_macro),
        'recall': np.array(avg_recall_macro),
        'precision_0': np.array(precision_0),
        'recall_0': np.array(recall_0),
        'precision_1': np.array(precision_1),
        'recall_1': np.array(recall_1),
        'auroc': np.array(auroc),
        'pr_auc': np.array(pr_auc)
    }
    
    return metrics