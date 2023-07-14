import pandas as pd
from log import *
from parser import *
from eval import *
from utils import *
from train import *
from module import NAT
import resource
import torch.nn as nn
import statistics

# import our benchmark library: benchtemp
import benchtemp as bt

args, sys_argv = get_args()

BATCH_SIZE = args.bs
NUM_NEIGHBORS = args.n_degree
NUM_EPOCH = args.n_epoch
ATTN_NUM_HEADS = args.attn_n_head
DROP_OUT = args.drop_out
DATA = args.data
NUM_HOP = args.n_hop
LEARNING_RATE = args.lr
POS_DIM = args.pos_dim
TOLERANCE = args.tolerance
VERBOSITY = args.verbosity
SEED = args.seed
TIME_DIM = args.time_dim
REPLACE_PROB = args.replace_prob
SELF_DIM = args.self_dim
NGH_DIM = args.ngh_dim
GPU = args.gpu
assert (NUM_HOP < 3)  # only up to second hop is supported
set_random_seed(SEED)
logger, get_checkpoint_path, get_ngh_store_path, get_self_rep_path, get_prev_raw_path, best_model_path, best_model_ngh_store_path = set_up_logger_nc(
    args, sys_argv)

# node_features, edge_features, full_data, train_data, val_data, test_data, new_node_val_data, \
#     new_node_test_data, new_old_node_val_data, new_old_node_test_data, new_new_node_val_data, \
#     new_new_node_test_data, unseen_nodes_num = get_data(DATA)

dataloader = bt.nc.DataLoader(dataset_path="./data/", dataset_name=DATA)
full_data, node_features, edge_features, train_data, val_data, test_data = dataloader.load()

e_feat = edge_features
n_feat = node_features

# src_l = full_data.sources
# dst_l = full_data.destinations
# e_idx_l = full_data.edge_idxs
# label_l = full_data.labels
# ts_l = full_data.timestamps
max_idx = max(full_data.sources.max(), full_data.destinations.max())

# split data according to the mask
train_src_l, train_dst_l, train_ts_l, train_e_idx_l, train_label_l = train_data.sources, train_data.destinations, train_data.timestamps, train_data.edge_idxs, train_data.labels
val_src_l, val_dst_l, val_ts_l, val_e_idx_l, val_label_l = val_data.sources, val_data.destinations, val_data.timestamps, val_data.edge_idxs, val_data.labels
test_src_l, test_dst_l, test_ts_l, test_e_idx_l, test_label_l = test_data.sources, test_data.destinations, test_data.timestamps, test_data.edge_idxs, test_data.labels

train_data = train_src_l, train_dst_l, train_ts_l, train_e_idx_l, train_label_l
val_data = val_src_l, val_dst_l, val_ts_l, val_e_idx_l, val_label_l
train_val_data = (train_data, val_data)

train_rand_sampler = bt.lp.RandEdgeSampler(train_src_l, train_dst_l)
val_rand_sampler = bt.lp.RandEdgeSampler(np.concatenate((train_src_l, val_src_l)), np.concatenate((train_dst_l, val_dst_l)), seed=0)
test_rand_sampler = bt.lp.RandEdgeSampler(np.concatenate((train_src_l, val_src_l, test_src_l)), np.concatenate((train_dst_l, val_dst_l, test_dst_l)), seed=1)
rand_samplers = train_rand_sampler, val_rand_sampler

# multiprocessing memory setting
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (200 * args.bs, rlimit[1]))

# model initialization

feat_dim = n_feat.shape[1]
e_feat_dim = e_feat.shape[1]
time_dim = TIME_DIM
model_dim = feat_dim + e_feat_dim + time_dim
hidden_dim = e_feat_dim + time_dim
num_raw = 3
memory_dim = NGH_DIM + num_raw
num_neighbors = [1]
for i in range(NUM_HOP):
    num_neighbors.extend([int(NUM_NEIGHBORS[i])])
# num_neighbors.extend([int(n) for n in NUM_NEIGHBORS]) # the 0-hop neighborhood has only 1 node

device = torch.device('cuda:{}'.format(GPU) if torch.cuda.is_available() else 'cpu')
test_auc_list = []
test_ap_list = []
nn_test_auc_list = []
nn_test_ap_list = []
new_old_test_auc_list = []
new_old_test_ap_list = []
new_new_test_auc_list = []
new_new_test_ap_list = []
for run in range(args.n_runs):
    total_start = time.time()
    nat = NAT(n_feat, e_feat, memory_dim, max_idx + 1, time_dim=TIME_DIM, pos_dim=POS_DIM, n_head=ATTN_NUM_HEADS,
              num_neighbors=num_neighbors, dropout=DROP_OUT,
              linear_out=args.linear_out, get_checkpoint_path=get_checkpoint_path,
              get_ngh_store_path=get_ngh_store_path, get_self_rep_path=get_self_rep_path,
              get_prev_raw_path=get_prev_raw_path, verbosity=VERBOSITY,
              n_hops=NUM_HOP, replace_prob=REPLACE_PROB, self_dim=SELF_DIM, ngh_dim=NGH_DIM, device=device)
    nat.to(device)
    nat.reset_store()

    optimizer = torch.optim.Adam(nat.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.BCELoss()
    early_stopper = bt.EarlyStopMonitor(tolerance=TOLERANCE)

    # start train and val phases
    train_val_nc(train_val_data, nat, args.mode, BATCH_SIZE, NUM_EPOCH, criterion, optimizer, early_stopper, rand_samplers,
              logger, model_dim, n_hop=NUM_HOP)

    test_auc = eval_one_epoch_nc('test for {} nodes'.format(args.mode), nat, test_rand_sampler,
                                                          test_src_l, test_dst_l, test_ts_l, test_label_l,
                                                          test_e_idx_l)

    # test_end = time.time()
    logger.info(
        'Test statistics: -- auc: {}'.format(test_auc))
    test_auc_list.append(test_auc)

    logger.info('Saving NAT model ...')
    torch.save(nat.state_dict(), best_model_path)
    logger.info('NAT model saved')

logger.info(
    'AVG+STD Test statistics: -- auc: {} \u00B1 {}'.format(
        np.average(test_auc_list), np.std(test_auc_list)))

logger.info("--------------Rounding to four decimal places--------------")
logger.info(
    'AVG+STD Test statistics: -- auc: {} \u00B1 {}'.format(
        np.around(np.average(test_auc_list), 4), np.around(np.std(test_auc_list), 4)))

