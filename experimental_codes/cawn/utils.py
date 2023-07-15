import numpy as np
import torch
import os
import random
import argparse
import sys
import pandas as pd


def get_args():
    parser = argparse.ArgumentParser(
        'Interface for Inductive Dynamic Representation Learning for Link Prediction on Temporal Graphs')

    # select dataset and training mode
    # choices = ['wikipedia', 'reddit', 'socialevolve', 'uci', 'enron', 'socialevolve_1month', 'socialevolve_2weeks'],
    parser.add_argument('-d', '--data', type=str, help='data sources to use, try wikipedia or reddit',
                        default='wikipedia')
    parser.add_argument('--data_usage', default=1.0, type=float, help='fraction of data to use (0-1)')
    parser.add_argument('-m', '--mode', type=str, default='i', choices=['t', 'i'],
                        help='transductive (t) or inductive (i)')
    parser.add_argument('--n_runs', type=int, default=1, help='Number of runs')
    # method-related hyper-parameters
    parser.add_argument('--n_degree', nargs='*', default=['64', '1'],
                        help='a list of neighbor sampling numbers for different hops, when only a single element is input n_layer will be activated')
    parser.add_argument('--n_layer', type=int, default=2, help='number of network layers')
    parser.add_argument('--bias', default=0.0, type=float,
                        help='the hyperparameter alpha controlling sampling preference in recent time, default to 0 which is uniform sampling')
    parser.add_argument('--agg', type=str, default='walk', choices=['tree', 'walk'],
                        help='tree based hierarchical aggregation or walk-based flat lstm aggregation, we only use the default here')
    parser.add_argument('--pos_enc', type=str, default='lp', choices=['spd', 'lp', 'saw'],
                        help='way to encode distances, shortest-path distance or landing probabilities, or self-based anonymous walk (baseline)')
    parser.add_argument('--pos_dim', type=int, default=172, help='dimension of the positional embedding')
    parser.add_argument('--pos_sample', type=str, default='binary', choices=['multinomial', 'binary'],
                        help='two equivalent sampling method with empirically different running time')
    parser.add_argument('--walk_pool', type=str, default='attn', choices=['attn', 'sum'],
                        help='how to pool the encoded walks, using attention or simple sum, if sum will overwrite all the other walk_ arguments')
    parser.add_argument('--walk_n_head', type=int, default=8, help="number of heads to use for walk attention")
    parser.add_argument('--walk_mutual', action='store_true',
                        help="whether to do mutual query for source and target node random walks")
    parser.add_argument('--walk_linear_out', action='store_true', default=False,
                        help="whether to linearly project each node's embedding")

    parser.add_argument('--attn_agg_method', type=str, default='attn', choices=['attn', 'lstm', 'mean'],
                        help='local aggregation method, we only use the default here')
    parser.add_argument('--attn_mode', type=str, default='prod', choices=['prod', 'map'],
                        help='use dot product attention or mapping based, we only use the default here')
    parser.add_argument('--attn_n_head', type=int, default=2,
                        help='number of heads used in tree-shaped attention layer, we only use the default here')
    parser.add_argument('--time', type=str, default='time', choices=['time', 'pos', 'empty'],
                        help='how to use time information, we only use the default here')

    # general training hyper-parameters
    parser.add_argument('--n_epoch', type=int, default=50, help='number of epochs')
    parser.add_argument('--bs', type=int, default=64, help='batch_size')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--drop_out', type=float, default=0.1, help='dropout probability for all dropout layers')
    parser.add_argument('--tolerance', type=float, default=0,
                        help='tolerated marginal improvement for early stopper')

    # parameters controlling computation settings but not affecting results in general
    parser.add_argument('--seed', type=int, default=0, help='random seed for all randomized algorithms')
    parser.add_argument('--ngh_cache', action='store_true',
                        help='(currently not suggested due to overwhelming memory consumption) cache temporal neighbors previously calculated to speed up repeated lookup')
    parser.add_argument('--gpu', type=int, default=0, help='which gpu to use')
    parser.add_argument('--cpu_cores', type=int, default=1, help='number of cpu_cores used for position encoding')
    parser.add_argument('--verbosity', type=int, default=1, help='verbosity of the program output')

    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)
    return args, sys.argv


class EarlyStopMonitor(object):
    def __init__(self, max_round=3, higher_better=True, tolerance=1e-3):
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


class RandEdgeSampler(object):
    def __init__(self, src_list, dst_list, seed=None):
        self.seed = None
        self.neg_sample = 'rnd'
        src_list = np.concatenate(src_list)
        dst_list = np.concatenate(dst_list)
        self.src_list = np.unique(src_list)
        self.dst_list = np.unique(dst_list)

        if seed is not None:
            self.seed = seed
            self.random_state = np.random.RandomState(self.seed)

    def sample(self, size):
        if self.seed is None:
            src_index = np.random.randint(0, len(self.src_list), size)
            dst_index = np.random.randint(0, len(self.dst_list), size)
        else:
            src_index = self.random_state.randint(0, len(self.src_list), size)
            dst_index = self.random_state.randint(0, len(self.dst_list), size)
        return self.src_list[src_index], self.dst_list[dst_index]

    def reset_random_state(self):
        self.random_state = np.random.RandomState(self.seed)


# class RandEdgeSampler(object):
#     def __init__(self, src_list, dst_list):
#         src_list = np.concatenate(src_list)
#         dst_list = np.concatenate(dst_list)
#         self.src_list = np.unique(src_list)
#         self.dst_list = np.unique(dst_list)
#
#     def sample(self, size):
#         src_index = np.random.randint(0, len(self.src_list), size)
#         dst_index = np.random.randint(0, len(self.dst_list), size)
#         return self.src_list[src_index], self.dst_list[dst_index]


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def process_sampling_numbers(num_neighbors, num_layers):
    num_neighbors = [int(n) for n in num_neighbors]
    if len(num_neighbors) == 1:
        num_neighbors = num_neighbors * num_layers
    else:
        num_layers = len(num_neighbors)
    return num_neighbors, num_layers


def get_data(dataset_name, different_new_nodes_between_val_and_test=False, randomize_features=False):
    ### Load data and train val test split
    graph_df = pd.read_csv('./data/ml_{}.csv'.format(dataset_name))
    edge_features = np.load('./data/ml_{}.npy'.format(dataset_name))
    node_features = np.load('./data/ml_{}_node.npy'.format(dataset_name))

    if randomize_features:
        node_features = np.random.rand(node_features.shape[0], node_features.shape[1])

    val_time, test_time = list(np.quantile(graph_df.ts, [0.70, 0.85]))

    sources = graph_df.u.values
    destinations = graph_df.i.values
    edge_idxs = graph_df.idx.values
    labels = graph_df.label.values
    timestamps = graph_df.ts.values

    full_data = Data(sources, destinations, timestamps, edge_idxs, labels)

    random.seed(2020)

    node_set = set(sources) | set(destinations)
    n_total_unique_nodes = len(node_set)

    # Compute nodes which appear at test time
    test_node_set = set(sources[timestamps > val_time]).union(
        set(destinations[timestamps > val_time]))
    # Sample nodes which we keep as new nodes (to test inductiveness), so than we have to remove all
    # their edges from training
    new_test_node_set = set(random.sample(test_node_set, int(0.1 * n_total_unique_nodes)))

    # Mask saying for each source and destination whether they are new test nodes
    new_test_source_mask = graph_df.u.map(lambda x: x in new_test_node_set).values
    new_test_destination_mask = graph_df.i.map(lambda x: x in new_test_node_set).values

    # Mask which is true for edges with both destination and source not being new test nodes (because
    # we want to remove all edges involving any new test node)
    observed_edges_mask = np.logical_and(~new_test_source_mask, ~new_test_destination_mask)

    # For train we keep edges happening before the validation time which do not involve any new node
    # used for inductiveness
    train_mask = np.logical_and(timestamps <= val_time, observed_edges_mask)

    train_data = Data(sources[train_mask], destinations[train_mask], timestamps[train_mask],
                      edge_idxs[train_mask], labels[train_mask])

    # define the new nodes sets for testing inductiveness of the model
    train_node_set = set(train_data.sources).union(train_data.destinations)
    assert len(train_node_set & new_test_node_set) == 0
    new_node_set = node_set - train_node_set

    val_mask = np.logical_and(timestamps <= test_time, timestamps > val_time)
    test_mask = timestamps > test_time

    if different_new_nodes_between_val_and_test:
        n_new_nodes = len(new_test_node_set) // 2
        val_new_node_set = set(list(new_test_node_set)[:n_new_nodes])
        test_new_node_set = set(list(new_test_node_set)[n_new_nodes:])

        edge_contains_new_val_node_mask = np.array(
            [(a in val_new_node_set or b in val_new_node_set) for a, b in zip(sources, destinations)])
        edge_contains_new_test_node_mask = np.array(
            [(a in test_new_node_set or b in test_new_node_set) for a, b in zip(sources, destinations)])
        new_node_val_mask = np.logical_and(val_mask, edge_contains_new_val_node_mask)
        new_node_test_mask = np.logical_and(test_mask, edge_contains_new_test_node_mask)


    else:
        # edge: new-
        edge_contains_new_node_mask = np.array(
            [(a in new_node_set or b in new_node_set) for a, b in zip(sources, destinations)])
        new_node_val_mask = np.logical_and(val_mask, edge_contains_new_node_mask)
        new_node_test_mask = np.logical_and(test_mask, edge_contains_new_node_mask)

        # edge: new-new
        edge_contains_new_new_node_mask = np.array(
            [(a in new_node_set and b in new_node_set) for a, b in zip(sources, destinations)])
        new_new_node_val_mask = np.logical_and(val_mask, edge_contains_new_new_node_mask)
        new_new_node_test_mask = np.logical_and(test_mask, edge_contains_new_new_node_mask)
        # way 2: anther way of constructing egde: new-old
        '''
        new- | new-new | new-old
        True | True    | False 
        True | False   | True 
        False| False   | False
        i.e. 
        new-old = new- && !new-new
        new-new = new- && !new-old
        new- = new-new || new-old
        '''
        edge_contains_new_old_node_mask = np.logical_and(edge_contains_new_node_mask,
                                                         np.logical_not(edge_contains_new_new_node_mask))
        new_old_node_val_mask = np.logical_and(val_mask, edge_contains_new_old_node_mask)
        new_old_node_test_mask = np.logical_and(test_mask, edge_contains_new_old_node_mask)

    # validation and test with all edges
    val_data = Data(sources[val_mask], destinations[val_mask], timestamps[val_mask],
                    edge_idxs[val_mask], labels[val_mask])

    test_data = Data(sources[test_mask], destinations[test_mask], timestamps[test_mask],
                     edge_idxs[test_mask], labels[test_mask])

    # validation and test with edges that at least has one new node (not in training set)
    new_node_val_data = Data(sources[new_node_val_mask], destinations[new_node_val_mask],
                             timestamps[new_node_val_mask],
                             edge_idxs[new_node_val_mask], labels[new_node_val_mask])

    new_node_test_data = Data(sources[new_node_test_mask], destinations[new_node_test_mask],
                              timestamps[new_node_test_mask], edge_idxs[new_node_test_mask],
                              labels[new_node_test_mask])

    # dataset:  new-old
    new_old_node_val_data = Data(sources[new_old_node_val_mask], destinations[new_old_node_val_mask],
                                 timestamps[new_old_node_val_mask],
                                 edge_idxs[new_old_node_val_mask], labels[new_old_node_val_mask])

    new_old_node_test_data = Data(sources[new_old_node_test_mask], destinations[new_old_node_test_mask],
                                  timestamps[new_old_node_test_mask], edge_idxs[new_old_node_test_mask],
                                  labels[new_old_node_test_mask])
    # dataset:  new-new
    new_new_node_val_data = Data(sources[new_new_node_val_mask], destinations[new_new_node_val_mask],
                                 timestamps[new_new_node_val_mask],
                                 edge_idxs[new_new_node_val_mask], labels[new_new_node_val_mask])

    new_new_node_test_data = Data(sources[new_new_node_test_mask], destinations[new_new_node_test_mask],
                                  timestamps[new_new_node_test_mask], edge_idxs[new_new_node_test_mask],
                                  labels[new_new_node_test_mask])

    # print("The dataset has {} interactions, involving {} different nodes".format(full_data.n_interactions,
    #                                                                              full_data.n_unique_nodes))
    # print("The training dataset has {} interactions, involving {} different nodes".format(
    #     train_data.n_interactions, train_data.n_unique_nodes))
    # print("The validation dataset has {} interactions, involving {} different nodes".format(
    #     val_data.n_interactions, val_data.n_unique_nodes))
    # print("The test dataset has {} interactions, involving {} different nodes".format(
    #     test_data.n_interactions, test_data.n_unique_nodes))
    # print("The new node validation dataset has {} interactions, involving {} different nodes".format(
    #     new_node_val_data.n_interactions, new_node_val_data.n_unique_nodes))
    # print("The new node test dataset has {} interactions, involving {} different nodes".format(
    #     new_node_test_data.n_interactions, new_node_test_data.n_unique_nodes))
    # unseen nodes num
    unseen_nodes_num = len(new_test_node_set)
    # print("{} nodes were used for the inductive testing, i.e. are never seen during training".format(
    #     unseen_nodes_num))

    return node_features, edge_features, full_data, train_data, val_data, test_data, \
        new_node_val_data, new_node_test_data, new_old_node_val_data, new_old_node_test_data, new_new_node_val_data, \
        new_new_node_test_data, unseen_nodes_num


def get_data_node_classification(dataset_name, use_validation=True):
    ### Load data and train val test split
    graph_df = pd.read_csv('./data/ml_{}.csv'.format(dataset_name))
    edge_features = np.load('./data/ml_{}.npy'.format(dataset_name))
    node_features = np.load('./data/ml_{}_node.npy'.format(dataset_name))

    val_time, test_time = list(np.quantile(graph_df.ts, [0.70, 0.85]))

    sources = graph_df.u.values
    destinations = graph_df.i.values
    edge_idxs = graph_df.idx.values
    labels = graph_df.label.values
    timestamps = graph_df.ts.values

    random.seed(2020)

    train_mask = timestamps <= val_time if use_validation else timestamps <= test_time
    test_mask = timestamps > test_time
    val_mask = np.logical_and(timestamps <= test_time, timestamps > val_time) if use_validation else test_mask

    full_data = Data(sources, destinations, timestamps, edge_idxs, labels)

    train_data = Data(sources[train_mask], destinations[train_mask], timestamps[train_mask],
                      edge_idxs[train_mask], labels[train_mask])

    val_data = Data(sources[val_mask], destinations[val_mask], timestamps[val_mask],
                    edge_idxs[val_mask], labels[val_mask])

    test_data = Data(sources[test_mask], destinations[test_mask], timestamps[test_mask],
                     edge_idxs[test_mask], labels[test_mask])

    return full_data, node_features, edge_features, train_data, val_data, test_data

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
