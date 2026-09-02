import torch
from sklearn.metrics import average_precision_score, roc_auc_score


def get_node_classification_metrics(predicts: torch.Tensor, labels: torch.Tensor):
    """
    get metrics for the node classification task
    :param predicts: Tensor, shape (num_samples, )
    :param labels: Tensor, shape (num_samples, )
    :return:
        dictionary of metrics {'metric_name_1': metric_1, ...}
    """
    predicts = predicts.cpu().detach().numpy()
    labels = labels.cpu().numpy()

    roc_auc = roc_auc_score(y_true=labels, y_score=predicts)
    pr_auc = average_precision_score(y_true=labels, y_score=predicts)

    return {'roc_auc': roc_auc, 'pr_auc': pr_auc}
