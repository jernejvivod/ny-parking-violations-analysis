import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay


def plot_roc(scores, y_test, pos_label, plot_path: str, file_name_for_plot: str):
    # Get false positive rates, true positive rates and thresholds.
    fpr, tpr, thresholds = metrics.roc_curve(y_test, scores[:, 1], pos_label=pos_label)

    # Compute AUC.
    roc_auc = metrics.roc_auc_score(y_test, scores[:, 1])

    # Plot ROC curve.
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = {0:4f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    plt.legend(loc="lower right")
    plt.savefig(plot_path + '/' + file_name_for_plot + '.png')
    plt.clf()
    plt.close()


def plot_confusion_matrix(predictions, y_test, labels, class_names, plot_path: str, file_name_for_plot):
    # Plot confusion matrix and save plot.
    np.set_printoptions(precision=2)
    disp = ConfusionMatrixDisplay.from_predictions(
        labels=labels,
        display_labels=class_names,
        y_true=y_test,
        y_pred=predictions,
        normalize='true',
        xticks_rotation='vertical'
    )

    disp.plot(cmap=plt.cm.Blues)
    plt.tight_layout()
    plt.savefig(plot_path + '/' + file_name_for_plot + '.png')
    plt.clf()
    plt.close()
