import numpy as np
import pandas as pd
import random

from benchtemp.utils import Data


class DataLoader:
    def __init__(self, dataset_path="./data/", dataset_name='mooc', use_validation=False):
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
        self.use_validation = use_validation

    def load(self):
        # Load data and train val test split
        graph_df = pd.read_csv(self.dataset_path + 'ml_{}.csv'.format(self.dataset_name))
        edge_features = np.load(self.dataset_path + 'ml_{}.npy'.format(self.dataset_name))
        node_features = np.load(self.dataset_path + 'ml_{}_node.npy'.format(self.dataset_name))

        val_time, test_time = list(np.quantile(graph_df.ts, [0.70, 0.85]))

        sources = graph_df.u.values
        destinations = graph_df.i.values
        edge_idxs = graph_df.idx.values
        labels = graph_df.label.values
        timestamps = graph_df.ts.values

        random.seed(2020)

        train_mask = timestamps <= val_time if self.use_validation else timestamps <= test_time
        test_mask = timestamps > test_time
        val_mask = np.logical_and(timestamps <= test_time, timestamps > val_time) if self.use_validation else test_mask

        full_data = Data(sources, destinations, timestamps, edge_idxs, labels)

        train_data = Data(sources[train_mask], destinations[train_mask], timestamps[train_mask],
                          edge_idxs[train_mask], labels[train_mask])

        val_data = Data(sources[val_mask], destinations[val_mask], timestamps[val_mask],
                        edge_idxs[val_mask], labels[val_mask])

        test_data = Data(sources[test_mask], destinations[test_mask], timestamps[test_mask],
                         edge_idxs[test_mask], labels[test_mask])

        return full_data, node_features, edge_features, train_data, val_data, test_data

