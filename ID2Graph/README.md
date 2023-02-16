# Eliminating Label Leakage in Tree-based Vertical Federated Learning

## 1. Dependencies

- C++11 compatible compiler
- Boost 1.65 or later
- Python 3.7 or later

## 2. Usage

### 2.1. Build from source

```
pip install -e .
./script/build.sh
```

### 2.2. Download datasets

```
./script/download.sh
```

### 2.3. Run experiments

- Example

```
./script/run.sh -u result -d ucicreditcard -m r
```

- Basic Arguments

```
    -u : (str) path to the folder to save the final results.
    -d : (str) name of dataset.
    -m : (str) type of training algorithm. `r`: Random Forest, `x`: XGBoost
```

- Advanced Arguments

```
    -t : (str) path to the folder to save the temporary results.
    -z : (int) number of trials.
    -p : (int) number of parallelly executed experiments.

    -n : (int) number of data records sampled for training.
    -f : (float) ratio of features owned by the active party.
    -v : (float) ratio of features owned by the passive party. if v=-1, the ratio of local features will be 1 - f.
    -i : (int) setting of feature importance. -1: normal, 1: unbalance

    -r : (int) total number of rounds for training.
    -j : (int) minimum number of samples within a leaf.
    -h : (int) maximum depth.
    -a : (float) learning rate of XGBoost.

    -e : (float) coefficient of edge weight (tau in our paper).
    -k : (float) weight for community variables.
    -l : (int) maximum number of iterations of Louvain
    -x : (optional) baseline union attack

    -b : (float) epsilon of ID-LMID.
    -c : (int) number of completely secure rounds.
    -o : (float) epsilon of LP-MST.

    -g : (optional) draw the extracted graph.
    -q : (optional) draw trees as html files.
```
