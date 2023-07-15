import pandas as pd
from log import *
from eval import *
from utils import *
from train import *
# import numba
from module import CAWN
from graph import NeighborFinder
import resource

# import our benchmark library: benchtemp
import benchtemp as bt

args, sys_argv = get_args()

BATCH_SIZE = args.bs
NUM_NEIGHBORS = args.n_degree
NUM_EPOCH = args.n_epoch
ATTN_NUM_HEADS = args.attn_n_head
DROP_OUT = args.drop_out
GPU = args.gpu
USE_TIME = args.time
ATTN_AGG_METHOD = args.attn_agg_method
ATTN_MODE = args.attn_mode
DATA = args.data
NUM_LAYER = args.n_layer
LEARNING_RATE = args.lr
POS_ENC = args.pos_enc
POS_DIM = args.pos_dim
WALK_POOL = args.walk_pool
WALK_N_HEAD = args.walk_n_head
WALK_MUTUAL = args.walk_mutual if WALK_POOL == 'attn' else False
TOLERANCE = args.tolerance
CPU_CORES = args.cpu_cores
NGH_CACHE = args.ngh_cache
VERBOSITY = args.verbosity
AGG = args.agg
SEED = args.seed
assert (CPU_CORES >= -1)
set_random_seed(SEED)
logger, get_checkpoint_path, best_model_path = set_up_logger(args, sys_argv)

dataloader = bt.lp.DataLoader(dataset_path="./data/", dataset_name=DATA)

### Extract data for training, validation and testing
node_features, edge_features, full_data, train_data, val_data, test_data, new_node_val_data, \
    new_node_test_data, new_old_node_val_data, new_old_node_test_data, new_new_node_val_data, \
    new_new_node_test_data, unseen_nodes_num = dataloader.load()

e_feat = edge_features
n_feat = node_features

src_l = full_data.sources
dst_l = full_data.destinations
e_idx_l = full_data.edge_idxs
label_l = full_data.labels
ts_l = full_data.timestamps
max_idx = max(full_data.sources.max(), full_data.destinations.max())

train_src_l, train_dst_l, train_ts_l, train_e_idx_l, train_label_l = train_data.sources, train_data.destinations, train_data.timestamps, train_data.edge_idxs, train_data.labels
val_src_l, val_dst_l, val_ts_l, val_e_idx_l, val_label_l = val_data.sources, val_data.destinations, val_data.timestamps, val_data.edge_idxs, val_data.labels
test_src_l, test_dst_l, test_ts_l, test_e_idx_l, test_label_l = test_data.sources, test_data.destinations, test_data.timestamps, test_data.edge_idxs, test_data.labels

test_src_new_l, test_dst_new_l, test_ts_new_l, test_e_idx_new_l, test_label_new_l = new_node_test_data.sources, new_node_test_data.destinations, new_node_test_data.timestamps, new_node_test_data.edge_idxs, new_node_test_data.labels
test_src_new_old_l, test_dst_new_old_l, test_ts_new_old_l, test_e_idx_new_old_l, test_label_new_old_l = new_old_node_test_data.sources, new_old_node_test_data.destinations, new_old_node_test_data.timestamps, new_old_node_test_data.edge_idxs, new_old_node_test_data.labels
test_src_new_new_l, test_dst_new_new_l, test_ts_new_new_l, test_e_idx_new_new_l, test_label_new_new_l = new_new_node_test_data.sources, new_new_node_test_data.destinations, new_new_node_test_data.timestamps, new_new_node_test_data.edge_idxs, new_new_node_test_data.labels

train_data = train_src_l, train_dst_l, train_ts_l, train_e_idx_l, train_label_l
val_data = val_src_l, val_dst_l, val_ts_l, val_e_idx_l, val_label_l
train_val_data = (train_data, val_data)

# create two neighbor finders to handle graph extraction.
# for transductive mode all phases use full_ngh_finder, for inductive node train/val phases use the partial one
# while test phase still always uses the full one
full_adj_list = [[] for _ in range(max_idx + 1)]
for src, dst, eidx, ts in zip(src_l, dst_l, e_idx_l, ts_l):
    full_adj_list[src].append((dst, eidx, ts))
    full_adj_list[dst].append((src, eidx, ts))
full_ngh_finder = NeighborFinder(full_adj_list, bias=args.bias, use_cache=NGH_CACHE, sample_method=args.pos_sample)
partial_adj_list = [[] for _ in range(max_idx + 1)]
for src, dst, eidx, ts in zip(train_src_l, train_dst_l, train_e_idx_l, train_ts_l):
    partial_adj_list[src].append((dst, eidx, ts))
    partial_adj_list[dst].append((src, eidx, ts))
for src, dst, eidx, ts in zip(val_src_l, val_dst_l, val_e_idx_l, val_ts_l):
    partial_adj_list[src].append((dst, eidx, ts))
    partial_adj_list[dst].append((src, eidx, ts))
partial_ngh_finder = NeighborFinder(partial_adj_list, bias=args.bias, use_cache=NGH_CACHE,
                                    sample_method=args.pos_sample)
ngh_finders = partial_ngh_finder, full_ngh_finder

# create random samplers to generate train/val/test instances
train_rand_sampler = bt.lp.RandEdgeSampler(train_src_l, train_dst_l)
val_rand_sampler = bt.lp.RandEdgeSampler(np.concatenate((train_src_l, val_src_l)), np.concatenate((train_dst_l, val_dst_l)), seed=0)
test_rand_sampler = bt.lp.RandEdgeSampler(np.concatenate((train_src_l, val_src_l, test_src_l)), np.concatenate((train_dst_l, val_dst_l, test_dst_l)), seed=1)
rand_samplers = train_rand_sampler, val_rand_sampler

# multiprocessing memory setting
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (200 * args.bs, rlimit[1]))

# model initialization
device = torch.device('cuda:{}'.format(GPU))
test_auc_list = []
test_ap_list = []
nn_test_auc_list = []
nn_test_ap_list = []
new_old_test_auc_list = []
new_old_test_ap_list = []
new_new_test_auc_list = []
new_new_test_ap_list = []

for i in range(args.n_runs):
    cawn = CAWN(n_feat, e_feat, agg=AGG,
                num_layers=NUM_LAYER, use_time=USE_TIME, attn_agg_method=ATTN_AGG_METHOD, attn_mode=ATTN_MODE,
                n_head=ATTN_NUM_HEADS, drop_out=DROP_OUT, pos_dim=POS_DIM, pos_enc=POS_ENC,
                num_neighbors=NUM_NEIGHBORS, walk_n_head=WALK_N_HEAD, walk_mutual=WALK_MUTUAL,
                walk_linear_out=args.walk_linear_out, walk_pool=args.walk_pool,
                cpu_cores=CPU_CORES, verbosity=VERBOSITY, get_checkpoint_path=get_checkpoint_path)
    cawn.to(device)
    optimizer = torch.optim.Adam(cawn.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.BCELoss()
    early_stopper = bt.EarlyStopMonitor(tolerance=TOLERANCE)

    # start train and val phases
    # train_val(train_val_data, cawn, args.mode, BATCH_SIZE, NUM_EPOCH, criterion, optimizer, early_stopper, ngh_finders, rand_samplers, logger)
    # unpack the data, prepare for the training

    mode = args.mode
    bs = BATCH_SIZE
    epochs = NUM_EPOCH

    num_instance = len(train_src_l)
    num_batch = math.ceil(num_instance / bs)
    logger.info('num of training instances: {}'.format(num_instance))
    logger.info('num of batches per epoch: {}'.format(num_batch))
    idx_list = np.arange(num_instance)

    for epoch in range(epochs):
        if mode == 't':  # transductive
            cawn.update_ngh_finder(full_ngh_finder)
        elif mode == 'i':  # inductive
            cawn.update_ngh_finder(partial_ngh_finder)
        else:
            raise ValueError('training mode {} not found.'.format(mode))
        start_epoch = time.time()
        acc, ap, f1, auc, m_loss = [], [], [], [], []
        np.random.shuffle(idx_list)  # shuffle the training samples for every epoch
        logger.info('start {} epoch'.format(epoch))
        for k in tqdm(range(num_batch)):
            # generate training mini-batch
            s_idx = k * bs
            e_idx = min(num_instance - 1, s_idx + bs)
            if s_idx == e_idx:
                continue
            batch_idx = idx_list[s_idx:e_idx]
            src_l_cut, dst_l_cut = train_src_l[batch_idx], train_dst_l[batch_idx]
            ts_l_cut = train_ts_l[batch_idx]
            e_l_cut = train_e_idx_l[batch_idx]
            label_l_cut = train_label_l[batch_idx]  # currently useless since we are not predicting edge labels
            size = len(src_l_cut)
            src_l_fake, dst_l_fake = train_rand_sampler.sample(size)

            # feed in the data and learn from error
            optimizer.zero_grad()
            cawn.train()
            pos_prob, neg_prob = cawn.contrast(src_l_cut, dst_l_cut, dst_l_fake, ts_l_cut,
                                               e_l_cut)  # the core training code
            pos_label = torch.ones(size, dtype=torch.float, device=device, requires_grad=False)
            neg_label = torch.zeros(size, dtype=torch.float, device=device, requires_grad=False)
            loss = criterion(pos_prob, pos_label) + criterion(neg_prob, neg_label)
            loss.backward()
            optimizer.step()

            # collect training results
            with torch.no_grad():
                cawn.eval()
                pred_score = np.concatenate([pos_prob.cpu().detach().numpy(), neg_prob.cpu().detach().numpy()])
                pred_label = pred_score > 0.5
                true_label = np.concatenate([np.ones(size), np.zeros(size)])
                acc.append((pred_label == true_label).mean())
                ap.append(average_precision_score(true_label, pred_score))
                # f1.append(f1_score(true_label, pred_label))
                m_loss.append(loss.item())
                auc.append(roc_auc_score(true_label, pred_score))

        # validation phase use all information
        val_acc, val_ap, val_f1, val_auc = eval_one_epoch('val for {} nodes'.format(mode), cawn, val_rand_sampler,
                                                          val_src_l,
                                                          val_dst_l, val_ts_l, val_label_l, val_e_idx_l)

        total_epoch_time = time.time() - start_epoch
        logger.info('epoch: {} took {:.2f}s'.format(epoch, total_epoch_time))
        # logger.info('epoch: {}:'.format(epoch))
        logger.info('epoch mean loss: {}'.format(np.mean(m_loss)))
        logger.info('train acc: {}, val acc: {}'.format(np.mean(acc), val_acc))
        logger.info('train auc: {}, val auc: {}'.format(np.mean(auc), val_auc))
        logger.info('train ap: {}, val ap: {}'.format(np.mean(ap), val_ap))
        if epoch == 0:
            # save things for data anaysis
            checkpoint_dir = '/'.join(cawn.get_checkpoint_path(0).split('/')[:-1])
            cawn.ngh_finder.save_ngh_stats(checkpoint_dir)  # for data analysis
            cawn.save_common_node_percentages(checkpoint_dir)

        # early stop check and checkpoint saving
        if early_stopper.early_stop_check(val_ap):
            logger.info('No improvment over {} epochs, stop training'.format(early_stopper.max_round))
            logger.info(f'Loading the best model at epoch {early_stopper.best_epoch}')
            best_checkpoint_path = cawn.get_checkpoint_path(early_stopper.best_epoch)
            cawn.load_state_dict(torch.load(best_checkpoint_path))
            logger.info(f'Loaded the best model at epoch {early_stopper.best_epoch} for inference')
            cawn.eval()
            break
        else:
            torch.save(cawn.state_dict(), cawn.get_checkpoint_path(epoch))

        cawn.update_ngh_finder(
            full_ngh_finder)  # remember that testing phase should always use the full neighbor finder
        test_acc, test_ap, test_f1, test_auc = eval_one_epoch('Training --- test for {} nodes'.format(args.mode), cawn,
                                                              test_rand_sampler, test_src_l, test_dst_l, test_ts_l,
                                                              test_label_l, test_e_idx_l)
        # test_new_new_acc, test_new_new_ap, test_new_new_auc, test_new_old_acc, test_new_old_ap, test_new_old_auc = [-1]*6
        test_new_acc, test_new_ap, test_new_f1, test_new_auc = eval_one_epoch(
            'Training --- test for {} nodes'.format(args.mode),
            cawn, test_rand_sampler, test_src_new_l,
            test_dst_new_l, test_ts_new_l,
            test_label_new_l, test_e_idx_new_l)
        test_new_new_acc, test_new_new_ap, test_new_new_f1, test_new_new_auc = eval_one_epoch(
            'Training --- test for {} nodes'.format(args.mode), cawn, test_rand_sampler, test_src_new_new_l,
            test_dst_new_new_l,
            test_ts_new_new_l, test_label_new_new_l, test_e_idx_new_new_l)
        test_new_old_acc, test_new_old_ap, test_new_old_f1, test_new_old_auc = eval_one_epoch(
            'Training --- test for {} nodes'.format(args.mode), cawn, test_rand_sampler, test_src_new_old_l,
            test_dst_new_old_l,
            test_ts_new_old_l, test_label_new_old_l, test_e_idx_new_old_l)

        logger.info(
            'Training --- Transductive: Test statistics: {} Old nodes -- acc: {}, auc: {}, ap: {}'.format(args.mode,
                                                                                                          test_acc,
                                                                                                          test_auc,
                                                                                                          test_ap))
        logger.info(
            'Training --- Inductive: Test statistics: {} New- nodes ---- acc: {}, auc: {}, ap: {}'.format(args.mode,
                                                                                                          test_new_acc,
                                                                                                          test_new_auc,
                                                                                                          test_new_ap))
        logger.info('Training --- Inductive: Test statistics: {} New-Old nodes - acc: {}, auc: {}, ap: {}'.format(args.mode,
                                                                                                                  test_new_old_acc,
                                                                                                                  test_new_old_auc,
                                                                                                                  test_new_old_ap))
        logger.info('Training --- Inductive: Test statistics: {} New-New nodes - acc: {}, auc: {}, ap: {}'.format(args.mode,
                                                                                                                  test_new_new_acc,
                                                                                                                  test_new_new_auc,
                                                                                                                  test_new_new_ap))

    # final testing
    cawn.update_ngh_finder(full_ngh_finder)  # remember that testing phase should always use the full neighbor finder

    test_acc, test_ap, test_f1, test_auc = eval_one_epoch('test for {} nodes'.format(args.mode), cawn,
                                                          test_rand_sampler, test_src_l, test_dst_l, test_ts_l,
                                                          test_label_l, test_e_idx_l)
    # test_new_new_acc, test_new_new_ap, test_new_new_auc, test_new_old_acc, test_new_old_ap, test_new_old_auc = [-1]*6
    test_new_acc, test_new_ap, test_new_f1, test_new_auc = eval_one_epoch('test for {} nodes'.format(args.mode), cawn,
                                                                          test_rand_sampler, test_src_new_l,
                                                                          test_dst_new_l, test_ts_new_l,
                                                                          test_label_new_l, test_e_idx_new_l)
    test_new_new_acc, test_new_new_ap, test_new_new_f1, test_new_new_auc = eval_one_epoch(
        'test for {} nodes'.format(args.mode), cawn, test_rand_sampler, test_src_new_new_l, test_dst_new_new_l,
        test_ts_new_new_l, test_label_new_new_l, test_e_idx_new_new_l)
    test_new_old_acc, test_new_old_ap, test_new_old_f1, test_new_old_auc = eval_one_epoch(
        'test for {} nodes'.format(args.mode), cawn, test_rand_sampler, test_src_new_old_l, test_dst_new_old_l,
        test_ts_new_old_l, test_label_new_old_l, test_e_idx_new_old_l)

    logger.info(
        'Transductive: Test statistics: {} Old nodes -- acc: {}, auc: {}, ap: {}'.format(args.mode, test_acc, test_auc,
                                                                                         test_ap))
    logger.info(
        'Inductive: Test statistics: {} New- nodes ---- acc: {}, auc: {}, ap: {}'.format(args.mode, test_new_acc,
                                                                                         test_new_auc, test_new_ap))
    logger.info(
        'Inductive: Test statistics: {} New-Old nodes - acc: {}, auc: {}, ap: {}'.format(args.mode, test_new_old_acc,
                                                                                         test_new_old_auc,
                                                                                         test_new_old_ap))
    logger.info(
        'Inductive: Test statistics: {} New-New nodes - acc: {}, auc: {}, ap: {}'.format(args.mode, test_new_new_acc,
                                                                                         test_new_new_auc,
                                                                                         test_new_new_ap))

    nn_test_auc_list.append(test_new_auc)
    nn_test_ap_list.append(test_new_ap)
    new_old_test_auc_list.append(test_new_old_auc)
    new_old_test_ap_list.append(test_new_old_ap)
    new_new_test_auc_list.append(test_new_new_auc)
    new_new_test_ap_list.append(test_new_new_ap)
    test_auc_list.append(test_auc)
    test_ap_list.append(test_ap)

    # save model
    logger.info('Saving CAWN model ...')
    torch.save(cawn.state_dict(), best_model_path)
    logger.info('CAWN model saved')

logger.info(
    'AVG+STD Transductive: ---------------- Old  nodes -- auc: {} \u00B1 {}, ap: {} \u00B1 {}'.format(
        np.average(test_auc_list), np.std(test_auc_list), np.average(test_ap_list), np.std(test_ap_list)))
logger.info(
    'AVG+STD Test statistics: Inductive: -- New- nodes -- auc: {} \u00B1 {}, ap: {} \u00B1 {}'.format(
        np.average(nn_test_auc_list), np.std(nn_test_auc_list), np.average(nn_test_ap_list), np.std(nn_test_ap_list)))
logger.info(
    'AVG+STD Test statistics: Inductive: New-Old nodes -- auc: {} \u00B1 {}, ap: {} \u00B1 {}'.format(
        np.average(new_old_test_auc_list), np.std(new_old_test_auc_list), np.average(new_old_test_ap_list),
        np.std(new_old_test_ap_list)))
logger.info(
    'AVG+STD Test statistics: Inductive: New-New nodes -- auc: {} \u00B1 {}, ap: {} \u00B1 {}'.format(
        np.average(new_new_test_auc_list), np.std(new_new_test_auc_list), np.average(new_new_test_ap_list),
        np.std(new_new_test_ap_list)))

logger.info("--------------Rounding to four decimal places--------------")
logger.info(
    'AVG+STD Transductive: ---------------- Old  nodes -- auc: {} \u00B1 {}, ap: {} \u00B1 {}'.format(
        np.around(np.average(test_auc_list), 4), np.around(np.std(test_auc_list), 4),
        np.around(np.average(test_ap_list), 4), np.around(np.std(test_ap_list), 4)))
logger.info(
    'AVG+STD Test statistics: Inductive: -- New- nodes -- auc: {} \u00B1 {}, ap: {} \u00B1 {}'.format(
        np.around(np.average(nn_test_auc_list), 4), np.around(np.std(nn_test_auc_list), 4),
        np.around(np.average(nn_test_ap_list), 4), np.around(np.std(nn_test_ap_list), 4)))
logger.info(
    'AVG+STD Test statistics: Inductive: New-Old nodes -- auc: {} \u00B1 {}, ap: {} \u00B1 {}'.format(
        np.around(np.average(new_old_test_auc_list), 4), np.around(np.std(new_old_test_auc_list), 4),
        np.around(np.average(new_old_test_ap_list), 4), np.around(np.std(new_old_test_ap_list), 4)))
logger.info(
    'AVG+STD Test statistics: Inductive: New-New nodes -- auc: {} \u00B1 {}, ap: {} \u00B1 {}'.format(
        np.around(np.average(new_new_test_auc_list), 4), np.around(np.std(new_new_test_auc_list), 4),
        np.around(np.average(new_new_test_ap_list), 4), np.around(np.std(new_new_test_ap_list), 4)))
