from sklearn.metrics import roc_auc_score, average_precision_score
import torch


class Evaluator:
    # if LP task. task_name="LP", as for NC task. task_name="NC"
    def __init__(self, task_name: str = "LP"):
        self.task_name = task_name

    def eval(self, pred_score, true_label):
        if torch is not None and isinstance(true_label, torch.Tensor):
            true_label = true_label.detach().cpu().numpy()

        if torch is not None and isinstance(pred_score, torch.Tensor):
            pred_score = pred_score.detach().cpu().numpy()

        # For LP task, Evaluation Metrics are the Area Under the Receiver Operating
        # Characteristic Curve (ROC AUC) and average precision (AP).
        if self.task_name == "LP":
            return average_precision_score(true_label, pred_score), roc_auc_score(true_label, pred_score)
        # For NC task, Evaluation Metric is ROC AUC.
        if self.task_name == "LP":
            return roc_auc_score(true_label, pred_score)
