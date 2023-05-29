import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import torch


class Data:
    def __init__(self, sources, destinations, timestamps, edge_idxs, labels):
        self.sources = sources
        self.destinations = destinations
        self.timestamps = timestamps
        self.edge_idxs = edge_idxs
        self.labels = labels
        self.n_interactions = len(sources)
        self.unique_nodes = set(sources) | set(destinations)
        self.n_unique_nodes = len(self.unique_nodes)


class EarlyStopMonitor(object):
    def __init__(self, max_round=3, higher_better=True, tolerance=1e-10):
        self.max_round = max_round
        self.num_round = 0

        self.epoch_count = 0
        self.best_epoch = 0

        self.last_best = None
        self.higher_better = higher_better
        self.tolerance = tolerance

    def early_stop_check(self, curr_val):
        if not self.higher_better:
            curr_val *= -1
        if self.last_best is None:
            self.last_best = curr_val
        elif (curr_val - self.last_best) / np.abs(self.last_best) > self.tolerance:
            self.last_best = curr_val
            self.num_round = 0
            self.best_epoch = self.epoch_count
        else:
            self.num_round += 1

        self.epoch_count += 1

        return self.num_round >= self.max_round


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
        # For NC task, Evaluation Metric is average precision (AP).
        if self.task_name == "LP":
            return roc_auc_score(true_label, pred_score)
