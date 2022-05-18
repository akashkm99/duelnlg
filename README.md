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
For automatic evaluations, you also need to download nlgeval data:
```bash
nlg-eval --setup
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

