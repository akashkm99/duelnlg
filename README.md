# DuelNLG
This repository contains code for evaluating NLG Models as described in the following paper:  
>[Active Evaluation: Efficient NLG Evaluation with Few Pairwise Comparisons](http://homes.cs.washington.edu/~marcotcr/acl20_checklist.pdf)  
> Akash Kumar Mohankumar, Mitesh M. Khapra. 
> Association for Computational Linguistics (ACL), 2022

Table of Contents
=================

   * [DuelNLG](#duelnlg)
      * [Table of Contents](#table-of-contents)
      * [Installation](#installation)
      * [Experiments from Paper](#experiments-from-paper)
         * [Download and Prepare Data](#download-and-prepare-data) 
         * [Model Free Algorithms](#model-free-algorithms)
         * [Model Based Algorithms](#model-based-algorithms)

## Installation
From Source:  
```bash
git clone https://github.com/akashkm99/duelnlg.git
cd duelnlg
pip install -e .
```
To use automatic metrics, you may also need to download nlgeval data:
```bash
python ./scripts/download/nlg-eval --setup
```

## Experiments from Paper
Here, we describe the steps to replicate the experiments mentioned in the paper. 

### Download and Prepare Data
To download and preprocess the WMT 2016 datasets:
```bash
bash scripts/preprocess/wmt16.sh
```
All the processed data will be stored as .pkl files at ```data/wmt16/processed/```

For the WMT 2015 datasets, 
```bash
bash scripts/preprocess/wmt15.sh
```
### Model Free Algorithms

To perform experiments with model-free dueling bandits algorithms, use the ```duelnlg/duelpy/experiments/experiments.py``` script. It has the following arguments:

* `--feedback-config`: A json config that specifies the list of datasets and their parameters. Use ```configs/feedback/wmt_all.json``` to run on all 7 WMT datasets.
* `--algorithm-config`: Config file that specifies the dueling bandit algorithms and their parameters. Use ```configs/algorithm/rmed.json``` to run the RMED algorithm and refer to ```configs/algorithm/default_all.json``` for the default parameters for all algorithms.
* `--output-dir`: Directory to save the results. (Default: ./results/bandits) 
* `--num-runs`: The number of times each algorithm is run with different random seed (Default: 200) 
* `--random-seed`: The base random seed to use (Default: 42)

For example, to run all the dueling bandits algorithm (except: IF and PL which are quite slow) on the WMT 2016 tur->eng dataset with 50 runs use:
```bash
python duelnlg/duelpy/experiments/experiments.py \
          --feedback-config ./configs/feedback/wmt16_tur_eng.json \
          --algorithm-config ./configs/algorithm/default_all_no_if_pl.json \
          --num-runs 50 
```

### Model Based Algorithms 

#### Download Training and Validation Data

To use direct evaluation metrics, we need to tune a few hyperparameters (e.g. thresholds for the preference probabilities) on a validation set. For training any end-to-end metric for pairwise prediction, we would also require a training set. 

To create the train and validation datasets for WMT, we use data from WMT 2013 and 2014:
```
bash scripts/prepare_train_val/wmt.sh
```
#### Automatic Evaluation Metrics

To run the Bleurt model, you need to download the model checkpoint:

```bash
bash scripts/download/bleurt_ckpt.sh
```

To run automatic metrics and save the predictions, use the ```duelnlg/direct_eval/evaluation.py``` script. It has the following arguments:

* `--metrics-config`: A json config that specifies the list of automatic metrics and their parameters. Use `configs/metrics/bleurt.json` to use bleurt and refer to `configs/metrics/all.json` to run all metrics. 
* `--val-path` and `test-path`: CSV files with the validation (for tuning) and test datasets.  E.g. for WMT 2016, it's `./data/wmt13_14/processed/val.csv` and `data/wmt16/processed/wmt16-human-judgements.csv` respectively. 
* `--processed-dir`: Directory with the processed .pkl files. E.g. for WMT 2016, it's `data/wmt16/processed`
* `--ensemble`: Whether to perform mulitple model forward passes with dropout for uncertainity estimation. Applicable only for Bleurt (Default: False) 
* `--multiref`: Whether the dataset has multiple reference texts. (Default: True)

For example, to run the Bleurt metric on WMT 2016 datasets, use the following:
```bash
python duelnlg/direct_eval/evaluation.py \
          --metrics ./configs/metrics/bleurt.json \
          --val-path ./data/wmt13_14/processed/val.csv \
          --test-path ./data/wmt16/processed/wmt16-human-judgements.csv \
          --output-results ./results/metrics/bleurt.csv \
          --processed-dir ./data/wmt16/processed
```



