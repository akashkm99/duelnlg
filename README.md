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

