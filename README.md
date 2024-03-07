# BenchTemp: A General Benchmark for Evaluating Temporal Graph Neural Networks

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![General badge](https://img.shields.io/badge/PyPI-v1.1.1-green.svg)]([mailto:jonnyhuanghnu@gmail.com](https://pypi.org/project/benchtemp/))
[![General badge](https://img.shields.io/badge/BenchTemp-Team-purple.svg)](https://my-website-6gnpiaym0891702b-1257259254.tcloudbaseapp.com/)
[![General badge](https://img.shields.io/badge/Gmail-BenchTemp-yellow.svg)](mailto:jonnyhuanghnu@gmail.com)
[![General badge](https://img.shields.io/badge/Wuhan-University-deepgreen.svg)]()
[![General badge](https://img.shields.io/badge/eBay-deepgreen.svg)]()
[![General badge](https://img.shields.io/badge/ETH-Zürich-deepgreen.svg)]()
<!-- [![General badge](https://img.shields.io/badge/Twitter-BenchTemp-blue.svg)](https://twitter.com/qianghuangwhu) -->





<!-- [![PyPI version](https://d25lcipzij17d.cloudfront.net/badge.svg?id=py&r=r&ts=1683906897&type=6e&v=1.1.1&x2=0)]() -->

## Table of Contents
- [BenchTemp: A General Benchmark for Evaluating Temporal Graph Neural Networks](#benchtemp-a-general-benchmark-for-evaluating-temporal-graph-neural-networks)
  - [Table of Contents](#table-of-contents)
  - [*News!!!*](#news)
  - [Overview](#overview)
  - [BenchTemp Framework](#benchtemp-framework)
  - [BenchTemp Pipeline](#benchtemp-pipeline)
  - [BenchTemp Modules](#benchtemp-modules)
  - [Installation](#installation)
    - [Requirements](#requirements)
    - [PyPI install](#pypi-install)
  - [Usage Example](#usage-example)
    - [*Dynamic Link Prediction*](#dynamic-link-prediction)
    - [*Dynamic Node  Classification*](#dynamic-node--classification)
  - [BenchTemp Reference](#benchtemp-reference)
    - [*DataPreprocessor*](#datapreprocessor)
    - [*TemporalGraph*](#temporalgraph)
    - [*lp.DataLoader*](#lpdataloader)
    - [*lp.RandEdgeSampler*](#lprandedgesampler)
    - [*nc.DataLoader*](#ncdataloader)
    - [*EarlyStopMonitor*](#earlystopmonitor)
    - [*Evaluator*](#evaluator)
  - [Call for Contributions](#call-for-contributions)

---

## *News!!!*
<br>

- *5/1/2024 - The Myket Dataset used in paper (https://arxiv.org/abs/2308.06862) has been added to BenchTemp. Data preprocess code at preprocess/Myket.py*

- *2/9/2023 - All datasets have been hosted on the open-source platform zenodo (https://zenodo.org/) with a Digital Object Identifier (DOI) 10.5281/zenodo.8267846 (https://zenodo.org/record/8267846).*

- *20/8/2023 - We have added four large-scale datasets **(eBay-Large, DGraphFin-Large, YouTubeReddit-Large, and Taobao-Large)**.*
<!-- including **four large-scale datasets** (eBay-Large, DGraphFin, YouTubeReddit-Large, and Taobao-Large) -->
- *12/7/2023 - We have uploaded experimental codes in folder **experimental_codes**.*
- *25/6/2023 - We have updated [BenchTemp website](https://my-website-6gnpiaym0891702b-1257259254.tcloudbaseapp.com/benchtemp.html).*
- *24/6/2023 - We have updated the reference of BenchTemp on github.*

---

## Overview
<br>

**BenchTemp** is a general Benchmark Python Library for evaluating Temporal Graph Neural Networks (TGNNs) quickly and efficiently on various workloads. 
**BenchTemp** provides **Benchmark Datasets**, and unified pipelines (**DataPreprocessor, DataLoader EdgeSampler, Evaluator, EarlyStopMonitor, BenchTempLoss, BenchTempOptimizer, and Leaderboard**) for evaluating Temporal Graph Neural Networks on both link prediction task and node classification task.

- Paper - <a href="https://arxiv.org/abs/2308.16385" target="_blank">*BenchTemp: A General Benchmark for Evaluating Temporal Graph Neural Networks*</a>.
<!-- - The BenchTemp PyPI Website is <a href="https://pypi.org/project/benchtemp/" target="_blank">Here</a>.

- The GitHub of BenchTemp project is <a href="https://github.com/qianghuangwhu/benchtemp" target="_blank">Here</a>.  -->
- Datasets - https://zenodo.org/record/8267846
- Code - https://github.com/qianghuangwhu/benchtemp
- Leaderboards - https://my-website-6gnpiaym0891702b-1257259254.tcloudbaseapp.com/ 

<!-- - Datasets - <a href="https://drive.google.com/drive/folders/1HKSFGEfxHDlHuQZ6nK4SLCEMFQIOtzpz?usp=sharing" target="_blank">Here</a>.  -->
<!-- 
- The leaderboards website for Temporal Graph Neural Networks on both link prediction task and node classification task is <a href="https://my-website-6gnpiaym0891702b-1257259254.tcloudbaseapp.com/" target="_blank">Here</a>. -->
- The source codes for evaluating existing Temporal Graph Neural Networks based on BenchTemp are in folder *experimental_codes*.

---

## BenchTemp Framework
<br>
<div align="center">
<img src="img/framework.png" alt="Overview of BenchTemp" style="width:100%;" />
</div>

---

## BenchTemp Pipeline
<br>
<div align="center">
<img src="img/pipeline.png" alt="BenchTemp Pipeline" style="width:100%;" />
</div>

---

## BenchTemp Modules
<br>
<div align="center">
<img src="img/modules.png" alt="BenchTemp Pipeline" style="width:70%;" />
</div>

---



## Installation

### Requirements


Please ensure that you have installed the following dependencies:

- numpy >= 1.18.0
- pandas >= 1.2.0
- sklearn >= 0.20.0

### PyPI install

```bash
pip install benchtemp 
```

---

## Usage Example
After installing benchtemp PyPI library, you can evaluating your TGNN models on dynamic link prediction task and dynamic node  classification task easily and quickly. For example:

### *Dynamic Link Prediction*
benchtemp provides *lp.DataLoader, lp.RandEdgeSampler， EarlyStopMonitor, Evaluator* for dynamic link prediction task. Users can evaluating their TGNN models by those components provided by benchtemp. See ***train_link_prediction.py*** in folder [*experimental_codes*](https://github.com/qianghuangwhu/benchtemp/tree/master/experimental_codes) for details.

Example Framework:

```python
# please import our benchtemp library
import benchtemp as bt

# For example, if you are training , you should create a training  RandEdgeSampler based on the training dataset.
data = bt.lp.DataLoader(dataset_path="./data/", dataset_name='mooc')

# dataloader for dynamic link prediction task

node_features, edge_features, full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data, new_old_node_val_data, new_old_node_test_data, new_new_node_val_data, new_new_node_test_data, unseen_nodes_num = data.load()


train_rand_sampler = bt.lp.RandEdgeSampler(train_data.sources, train_data.destinations)

monitor = bt.EarlyStopMonitor()

# Users' own TGNN models or SOTA TGNN models
model = TGNN(parameters)
...
for epoch in range(args.epochs):
    ...
    # sample an equal amount of negatives to the positive interactions.
    size = len(train_data)
    _, negatives_batch = train_rand_sampler.sample(size)
    ...
    ...
    pre_positive, pre_negative = model(positive_batch,negatives_batch)
    loss = loss_function(pre_positive, pre_negative, labels)
    ...
    ...
    val_ap = model(val_data)
    if monitor.early_stop_check(val_ap):
      break
...

# testing
pre = model(test_data)
results = bt.evaluator(pre, labels)
```
### *Dynamic Node  Classification*
benchtemp provides *nc.DataLoader, EarlyStopMonitor, Evaluator* for dynamic node  classification. See ***train_node_classification.py*** in folder [*experimental_codes*](https://github.com/qianghuangwhu/benchtemp/tree/master/experimental_codes)   for details. 

```python
# please import our benchtemp library
import benchtemp as bt

# For example, if you are training , you should create a training  RandEdgeSampler based on the training dataset.
data = bt.nc.DataLoader(dataset_path="./data/", dataset_name='mooc')

# dataloader for dynamic node  classification task

node_features, edge_features, full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data, new_old_node_val_data, new_old_node_test_data, new_new_node_val_data, new_new_node_test_data, unseen_nodes_num = data.load()


# Users' own TGNN models or SOTA TGNN models
model = TGNN(parameters)
...
for epoch in range(args.epochs):
    ...
    # sample an equal amount of negatives to the positive interactions.
    size = len(train_data)
    ...
    ...
    pre_positive, pre_negative = model(positive_batch,negatives_batch)
    loss = loss_function(pre_positive, pre_negative, labels)
    ...
    ...
    val_ap = model(val_data)
    if monitor.early_stop_check(val_ap):
      break
...

# testing
pre = model(test_data)
results = bt.evaluator(pre, labels)
```

---

## BenchTemp Reference

---

### *DataPreprocessor*
<br>

The datasets that have been preprocessed by BenchTemp are [Here](https://drive.google.com/drive/folders/1HKSFGEfxHDlHuQZ6nK4SLCEMFQIOtzpz?usp=sharing).
You can directly download the datasets and then put them into the directory './data'.

In addition, BenchTemp provides *DataPreprocessor* class for you to preprocess yours TGNNs datasets. 
<!-- you can download the original datasets [Here]() and then  -->
<!-- use the functions provided by BenchTemp for data data. -->

*Class*:
```python
DataPreprocessor(data_path: str, data_name: str)
```

*Args*:

- *data_path: str* - The path of the dataset.
- *data_name: str* - The name of the dataset.


*Function*:
```python
DataPreprocessor.data_preprocess(bipartite: bool)
```

*Args*:

- *bipartite: bool* - Whether the Temporal Graph is a bipartite graph (*Heterogeneous* or *Homogeneous*).

*Returns*:

- *ml_{data_name}.csv* - The csv file of the Temporal Graph.
This file have five columns with properties:
   - '*u*': The id of the user.
   - '*i*': The id of the item.
   - '*ts*': The timestamp of the interaction (edge) between the user and the item.
   - '*label*': The label of the interaction (edge).
   - '*idx*': The index of the interaction (edge).

- *ml_{data_name}.npy* - The edge features corresponding to the interactions (edges) in the the Temporal Graph..


- *ml_{data_name}_node.npy* - The initialization node features of the Temporal Graph.

Example:

```python
import benchtemp as bt

processor = bt.DataPreprocessor(data_path="./data/", data_name="mooc")
# If the dataset is bipartite graph, i.e. the user (source nodes) and the item (destination nodes) are of the same type.
processor.data_preprocess(bipartite=True)

# If the dataset is non-bipartite graph.
processor.data_preprocess(bipartite=False)
```

<!-- Notes:

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
``` -->
---


### *TemporalGraph*
<br>

The class of a temporal graph. A temporal graph can  be represented as an ordered sequence of temporal user-item
interactions $I_{r}=(u_{r}, i_{r}, t_{r}, e_{r})$, $0 \leq t_{1} \leq  \dots  t_{r} \dots \leq  T$. The $r$-th interaction $I_{r}$ happens at time $t_{r}$ between user $u_{r}$ and item $i_{r}$
with edge feature $e_{r}$.

*Class*:
```python
TemporalGraph(sources: numpy.array, destinations: numpy.array, timestamps: numpy.array, edge_idxs: numpy.array, labels: numpy.array)
```

*Args*:
- *sources: numpy.array* - Array of sources of Temporal Graph edges.
- *destinations: numpy.array* - Array of destinations of Temporal Graph edges.
- *timestamps: numpy.array* - Array of timestamps of Temporal Graph edges.
- *edge_idxs: numpy.array* - Array of edge IDs of Temporal Graph edges.
- *labels: numpy.array* - Array of labels of Temporal Graphe dges.

*Returns*: 
- *benchtemp.TemporalGraph*. A Temporal Graph.




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
full_data = bt.TemporalGraph(sources, destinations, timestamps, edge_idxs, labels)
```

---

### *lp.DataLoader*

<br>

The DataLoader class for link prediction tasks. 

In *transductive link prediction*, Dataloader splits the temporal graphs chronologically into 70\%-15\%-15\% for train, validation and test sets according to edge timestamps. 

In *inductive link prediction*, Dataloader performs the same split as the transductive setting, and randomly masks 10\% nodes as unseen nodes.
Any edges associated with these unseen nodes are removed from the training set.
To reflect different inductive scenarios,
DataLoader further generates three inductive test sets from the transductive test dataset, by filtering edges in different manners:

  - **Inductive** -  selects edges with at least one unseen node. 
  - **Inductive New-Old** - selects edges between a seen node and an unseen node.
  - **Inductive New-New** - selects edges between two unseen nodes. 

*Class*:

```python
lp.DataLoader(dataset_path: str, dataset_name: str, different_new_nodes_between_val_and_test: bool, randomize_features: bool)
```

*Args*:

- *dataset_path: str* - The path of the dataset.
- *dataset_name: str* - The name of dataset.
- *different_new_nodes_between_val_and_test: bool* - The new nodes are between validation set and test set.
- *randomize_features: str* - Random initialization of node features. 

*Function*:

```python
lp.DataLoader.load()
```

*Returns*:
- *node_features: numpy.array* - Array of the Node Features of the Temporal Graph. 
- *edge_features: numpy.array* - Array of the Edge Features of the Temporal Graph.
- *full_data: benchtemp.TemporalGraph* - Full Temporal Graph dataset. 
- *train_data: benchtemp.TemporalGraph* - The training set. 
- *val_data: benchtemp.TemporalGraph* - The validation set.
- *test_data: benchtemp.TemporalGraph*  - The **Transductive** test set.
- *new_node_val_data: benchtemp.TemporalGraph* - The **Inductive**  validation set.
- *new_node_test_data: benchtemp.TemporalGraph* - The **Inductive**  test set.
- *new_old_node_val_data: benchtemp.TemporalGraph* - The **Inductive New-Old** validation set.
- *new_old_node_test_data: benchtemp.TemporalGraph* - The **Inductive New-Old**  test set.
- *new_new_node_val_data: benchtemp.TemporalGraph* - The **Inductive New-New** validation set.
- *new_new_node_test_data: benchtemp.TemporalGraph* - The **Inductive New-New** test set.
- *unseen_nodes_num: int* - The number of unseen nodes in inductive setting.

Example:

```python
import benchtemp as bt

data = bt.lp.DataLoader(dataset_path="./data/", dataset_name='mooc')

node_features, edge_features, full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data, new_old_node_val_data, new_old_node_test_data, new_new_node_val_data, new_new_node_test_data, unseen_nodes_num = data.load()
```

---

### *lp.RandEdgeSampler*

<br>

BenchTemp provides the unified
negative edge sampler class with a **seed** named RandEdgeSampler  for link prediction task to  sample an equal amount of negatives to the positive interactions.

*Class*:

```python
RandEdgeSampler(src_list: numpy.array, dst_list: numpy.array, seed: int)
```

*Args*:
- *src_list: numpy.array* - Array of source nodes.
- *dst_list: numpy.array* - Array of destination nodes.
- *seed: numpy.array* - The seed of random.

*Function*: 

```python
RandEdgeSampler.sample(size: int)
```

*Args*:
- *size: int* - The size of the sampling negative edges.

*Returns*:
- *src_list: numpy.array* - Array of source nodes of the sampling negative edges.
- *dst_list: numpy.array* - Array of destination nodes of the sampling negative edges.

Example:

```python
import benchtemp as bt

# For example, if you are training , you should create a training  RandEdgeSampler based on the training dataset.
train_rand_sampler = bt.lp.RandEdgeSampler(train_data.sources, train_data.destinations)

...
for epoch in range(args.epochs):
    ...
    # sample an equal amount of negatives to the positive interactions.
    size = len(train_data)
    _, negatives_batch = train_rand_sampler.sample(size)
    ...
...
```

---

<br>

### *nc.DataLoader*

<br>

The DataLoader class for the node classification task. The DataLoader module sorts edges and splits the input dataset (70\%-15\%-15\%) according to edge timestamps.

*Class*:

```python
nc.DataLoader(dataset_path: str, dataset_name: str, use_validation: bool)
```

*Args*:
- *dataset_path: str* - The path of the dataset.
- *dataset_name: str* - The name of the dataset.
- *use_validation: bool* - Whether use validation dataset or not.

*Function*:

```python
nc.DataLoader.load()
``` 

*Returns*:
- *node_features: numpy.array* - Array of the Node Features of the Temporal Graph. 
- *edge_features: numpy.array* - Array of the Edge Features of the Temporal Graph.
- *full_data: benchtemp.TemporalGraph* - Full Temporal Graph dataset for node classification task. 
- *train_data: benchtemp.TemporalGraph* - The training set for node classification task. 
- *val_data: benchtemp.TemporalGraph* - The validation set for node classification task.
- *test_data: benchtemp.TemporalGraph*  - The test set for node classification task.

Example:
```python
import benchtemp as bt

data = bt.nc.DataLoader(dataset_path="./data/", dataset_name='mooc', use_validation=True)

node_features, edge_features, full_data, train_data, val_data, test_data = data.load()
```

---
<br>

### *EarlyStopMonitor*

<br>

BenchTemp provides a unified EarlyStopMonitor to improve training efficiency and save resources.



*Class*:
```python
EarlyStopMonitor(max_round: int, higher_better: bool, tolerance: float)
```
*Args*:
- *max_round: int* - The number of rounds for early stop.
- *higher_better: bool* - The higher the value, the better the performance.
- *tolerance: float* - The tolerance of the EarlyStopMonitor.

*Function*:
```python
EarlyStopMonitor.early_stop_check(curr_val:float)
```
*Args*:

- *curr_val: float* - The value to check for early stop.

*Returns*:

- *True* - If the value matches the setting of the EarlyStopMonitor.
- *False* - If the value does not match the setting of the EarlyStopMonitor.

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

---

<br>

### *Evaluator*

<br>


Different evaluation metrics are available, including Area Under the Receiver Operating Characteristic Curve (ROC AUC) and Average Precision (AP). Usually, metrics *Area Under the Receiver Operating Characteristic Curve (ROC AUC)* and *average precision (AP)* are for the link prediction task, while metrics *AUC* is for the node classification task.
<!-- **link prediction** Evaluation Metrics  are **Area Under the Receiver Operating Characteristic Curve (ROC AUC)** and **average precision (AP)**

**node classification** Evaluation Metric is **Area Under the Receiver Operating Characteristic Curve (ROC AUC)** -->

*Class*: 
```python
Evaluator(task_name: str)
```

*Args*:
- *task_name: str*  - the name of the task, choice in **["LP", "NC"]**, LP for the link prediction task and NC for the node classification task.

*Function*:
```python
Evaluator.eval(pred_score: numpy.array, true_label: numpy.array)
```
*Args*:
- *pred_score: numpy.array*- Array of prediction scores.
- *true_label: numpy.array* - Array of true labels.

*Returns*:

- *AUC: float* - the value of the AUC.
- *AP: float* - the value of the AP.

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

# For example, node classification task. Evaluation Metrics: AUC.
evaluator = bt.Evaluator("NC")

...
# test data
pred_score = model(test_data)
test_auc = evaluator.eval(pred_score, true_label)
...
```
---

## Call for Contributions

<br>

BenchTemp project is looking for contributors with 
expertise and enthusiasm! If you have the desire to contribute to BenchTemp, 
please contact [BenchTemp team](mailto:jonnyhuanghnu@gmail.com). Contributions and issues from the community are eagerly welcomed, with which
we can together push forward the TGNN research.

