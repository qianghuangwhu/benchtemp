# BenchTemp: A General Benchmark for Evaluating Temporal Graph Neural Networks

## BenchTemp Framework
<img src="img/framework.png" alt="Overview of BenchTemp" style="width:100%;" />

## BenchTemp Pipeline
<img src="img/pipeline.png" alt="BenchTemp Pipeline" style="width:100%;" />

## Overview
**BenchTemp** is a general Benchmark Python Library for users to evaluate Temporal Graph Neural Networks (TGNNs) quickly and efficiently on various workloads. 
**BenchTemp** provides **Benchmark Datasets**, and unified pipelines (**DataPreprocessor, DataLoader EdgeSampler, Evaluator, EarlyStopMonitor, BCELoss, Optimizer,and Leaderboard** ) for evaluating Temporal Graph Neural Networks on both link prediction and node classification tasks.

- The BenchTemp PyPI Website is [Here](https://pypi.org/project/benchtemp/).
- The GitHub of BenchTemp project is [Here](https://github.com/qianghuangwhu/benchtemp).
- The datasets are [Here](https://drive.google.com/drive/folders/1HKSFGEfxHDlHuQZ6nK4SLCEMFQIOtzpz?usp=sharing).
- The leaderboards website for Temporal Graph Neural Networks on both Link Prediction and Node Classification tasks is [Here](https://my-website-6gnpiaym0891702b-1257259254.tcloudbaseapp.com/).
- The source codes for evaluating existing Temporal Graph Neural Networks based on BenchTemp are [coming soon](). 


## Installation
### Requirements
Please ensure that you have installed the following dependencies:

- numpy >= 1.18.0
- pandas >= 1.2.0
- sklearn >= 0.20.0

### BenchTemp PyPI install

```bash
pip3 install benchtemp 
```


## Package Usage


### Datasets
The datasets that have been preprocessed by BenchTemp are [Here](https://drive.google.com/drive/folders/1HKSFGEfxHDlHuQZ6nK4SLCEMFQIOtzpz?usp=sharing).
You can directly download the datasets and then put them into the directory './data'.

In addition, BenchTemp provides data processing functions for you to preprocess yours TGNNs datasets. 
<!-- you can download the original datasets [Here]() and then  -->
<!-- use the functions provided by BenchTemp for data data. -->

Function:
**benchtemp.preprocessing.data.data_preprocess(data_name : str, bipartite : bool=True)**

Parameters:
- **data_name : str** - the name of the dataset.
- **bipartite : bool** - Whether the Temporal Graph is bipartite graph.

Returns:
- **ml_{data_name}.csv** - the csv file of the Temporal Graph.
- **ml_{data_name}.npy** - the edge features of the Temporal Graph.
- **ml_{data_name}_node.npy** - the node features of the Temporal Graph.

Example:

```python
import benchtemp as bt

# If the dataset is bipartite graph, i.e. the user (source nodes) and the item (destination nodes) are of the same type.
bt.data_preprocess("data_name", bipartite=True)

# non-bipartite graph.
bt.data_preprocess("data_name", bipartite=False)
```

Notes:

For bipartite graph, BenchTemp will factorize the source node index and 
the destination node index, respectively. 
```python
import pandas as pd

assert len(sources) == len(destinations)

# bipartite graph
sources, _ = pd.factorize(sources)
destinations, _ = pd.factorize(destinations)

upper_u = sources.max + 1
destinations = destinations + upper_u
```
For non-bipartite graph, BenchTemp will factorize the concatenation of source node array and 
the destination node array. 

```python
import pandas as pd
import numpy as np

assert len(sources) == len(destinations)
interaction_num = len(sources)

# non-bipartite graph
node_index, _ = pd.factorize(np.concatenate((sources, destinations), axis=0))

sources = node_index[[0:interaction_num]]
destinations = node_index[[interaction_num:interaction_num + interaction_num]]
```


### TemporalData Class
Class:

**Data(sources : numpy.ndarray,
destinations : numpy.ndarray,
timestamps : numpy.ndarray,
edge_idxs : numpy.ndarray,
labels : numpy.ndarray)**

Parameters:
- **sources : numpy.ndarray** - Array of sources of Temporal Graph edges.
- **destinations : numpy.ndarray** - Array of destinations of Temporal Graph edges.
- **timestamps : numpy.ndarray** - Array of timestamps of Temporal Graph edges.
- **edge_idxs : numpy.ndarray** - Array of edge IDs of Temporal Graph edges.
- **labels : numpy.ndarray** - Array of labels of Temporal Graphe dges.

Returns: 
- **benchtemp.Data**. A Temporal Graph.




Example:
```python
import pandas as pd
import numpy as np
import benchtemp as bt


graph_df = pd.read_csv("dataset_path")

sources = graph_df.u.values
destinations = graph_df.i.values
edge_idxs = graph_df.idx.values
labels = graph_df.label.values
timestamps = graph_df.ts.values

# For example, the full Temporal Graph of the dataset is full_data.
full_data = bt.Data(sources, destinations, timestamps, edge_idxs, labels)
```


### DataLoader
### (1) Link Prediction task
Function:

**benchtemp.lp.readers.get_data(dataset_name : str, different_new_nodes_between_val_and_test=False, randomize_features=False)**
 
Parameters:
- **dataset_name : str** - The name of the dataset. The dataset file (.csv file of the Temporal Graph, .npy file of the node features and .npy file of the edge features) should be in "./data" directory.
- **different_new_nodes_between_val_and_test : bool** - Different new nodes between  the validation and test dataset.
- **randomize_features : bool** - Random initialization of node Features. 

Returns:
- **node_features : numpy.ndarray** - Array of the Node Features of the Temporal Graph. 
- **edge_features : numpy.ndarray** - Array of the Edge Features of the Temporal Graph.
- **full_data : benchtemp.Data** - Full Temporal Graph dataset. 
- **train_data : benchtemp.Data** - The training Temporal Graph dataset. 
- **val_data : benchtemp.Data** - The validation Temporal Graph dataset.
- **test_data : benchtemp.Data**  - The **transductive** test Temporal Graph dataset.
- **new_node_val_data : benchtemp.Data** - The **inductive [new-]** setting validation Temporal Graph dataset.
- **new_node_test_data : benchtemp.Data** - The **inductive [new-]** setting test Temporal Graph dataset.
- **new_old_node_val_data : benchtemp.Data** - The **inductive [new-old]** setting validation Temporal Graph dataset.
- **new_old_node_test_data : benchtemp.Data** - The **inductive [new-old]** setting test Temporal Graph dataset.
- **new_new_node_val_data : benchtemp.Data** - The **inductive [new-new]** setting validation Temporal Graph dataset.
- **new_new_node_test_data : benchtemp.Data** - The **inductive [new-new]** setting test Temporal Graph dataset.
- **unseen_nodes_num : int** - The number of unseen nodes of inductive setting.

Example:

```python
import benchtemp as bt

data = bt.lp.DataLoader(dataset_path="./data/", dataset_name='mooc')

node_features, edge_features, full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data, new_old_node_val_data, new_old_node_test_data, new_new_node_val_data, new_new_node_test_data, unseen_nodes_num = data.load()
```

### EdgeSampler
BenchTemp provides the unified
negative edge sampler with **seed** for Link Prediction task to  sample an equal amount of negatives to the positive interactions.

Class:

**RandEdgeSampler(src_list : numpy.ndarray, dst_list : numpy.ndarray, seed : int =None)**

Parameters:
- **src_list : numpy.ndarray** - the list of source nodes.
- **dst_list : numpy.ndarray** - the list of destination nodes.
- **seed : numpy.ndarray** - seed of random.

Returns: 

- **benchtemp.RandEdgeSampler**

Example:

```python
import benchtemp as bt

# For example, if you are training , you should create a training  RandEdgeSampler based on the training dataset.
train_rand_sampler = bt.lp.RandEdgeSampler(train_data.sources, train_data.destinations)

...
for epoch in range(args.epochs):
    ...
    # sample an equal amount of negatives to the positive interactions.
    _, negatives_batch = train_rand_sampler.sample(size)
    ...
...
```
### (2) Node Classification task
Function:



**benchtemp.nc.readers.get_data_node_classification((dataset_name : str, use_validation : bool=False))** 

Parameters:
- dataset_name : str - The name of the dataset. The dataset file (.csv file of the Temporal Graph, .npy file of the node features and .npy file of the edge features) should be in "./data" directory.
- use_validation : bool - Whether use validation dataset or not.

Returns:
- **node_features : numpy.ndarray** - Array of the Node Features of the Temporal Graph. 
- **edge_features : numpy.ndarray** - Array of the Edge Features of the Temporal Graph.
- **full_data : benchtemp.Data** - Full Temporal Graph dataset for Node Classification task. 
- **train_data : benchtemp.Data** - The training Temporal Graph dataset for Node Classification task. 
- **val_data : benchtemp.Data** - The validation Temporal Graph dataset for Node Classification task.
- **test_data : benchtemp.Data**  - The test Temporal Graph dataset for Node Classification task.



### EarlyStopMonitor
Class:

**EarlyStopMonitor(max_round=3, higher_better=True, tolerance=1e-10)**

Parameters:
- **max_round : int** - the maximum number of rounds of EarlyStopMonitor.
- **higher_better : bool** - better the performance.
- **tolerance : float** - the tolerance of the EarlyStopMonitor.

Returns:
- **benchtemp.EarlyStopMonitor**

Example:
```python
import benchtemp as bt

...
early_stopper = bt.EarlyStopMonitor(max_round=args.patience)
for epoch in range(args.epochs):
    ...
    val_ap = model(val_datasets)
    if early_stopper.early_stop_check(val_ap):
        break
    ...
...
```

### Evaluator

**Link Prediction** Evaluation Metrics  are **Area Under the Receiver Operating Characteristic Curve (ROC AUC)** and **average precision (AP)**

**Node Classification** Evaluation Metric is **Area Under the Receiver Operating Characteristic Curve (ROC AUC)**

Class: 

**Evaluator(task_name: str = "LP")**

Parameters:
- task_name : str  - the name of the task, choice in **["LP", "NC"]**.

Returns:
- **benchtemp.Evaluator**

Example:

```python
import benchtemp as bt

# For example, Link prediction task. Evaluation Metrics: AUC, AP.
evaluator = bt.Evaluator("LP")

...
# test data
pred_score = model(test_data)
test_auc, test_ap = evaluator.eval(pred_score, true_label)
...
```

```python
import benchtemp as bt

# For example, Node Classification task. Evaluation Metrics: AUC.
evaluator = bt.Evaluator("NC")

...
# test data
pred_score = model(test_data)
test_auc = evaluator.eval(pred_score, true_label)
...
```

## Call for Contributions

**BenchTemp** project is looking for contributors with 
expertise and enthusiasm! If you have a desire to contribute to **BenchTemp**, 
please contact [BenchTemp team](mailto:jonnyhuanghnu@gmail.com).

\item We release \sys, a general benchmark library for evaluating temporal graph models;

\item We provides unified benckmark temporal graph datasets processed by \sys;

\item We benchmark existing temporal graph models based on \sys  for robust developments and analysis the performance of models in detail;

\item We relsed the leaderboards for temporal graph models.

