rm -rf data/wmt13_14
mkdir -p data/wmt13_14

echo "Downloading WMT 2014 Data.."
wget https://www.statmt.org/wmt14/submissions.tgz -O data/wmt13_14/wmt14_submissions.tgz
wget https://www.statmt.org/wmt14/wmt14-manual-evaluation.tgz -O data/wmt13_14/wmt14-manual-evaluation.tgz
echo "Unzipping Files .."
tar zxvf data/wmt13_14/wmt14_submissions.tgz -C data/wmt13_14
tar zxvf data/wmt13_14/wmt14-manual-evaluation.tgz -C data/wmt13_14
echo "Preprocessing files .."
mkdir -p data/wmt13_14/processed
python preprocess/prepare_data_wmt13_14.py --raw-dir ./data/wmt13_14 --processed-dir ./data/wmt13_14/processed --year 2014

echo "Downloading WMT 2013 Data.."
wget https://www.statmt.org/wmt13/wmt13-data.tar.gz -O data/wmt13_14/wmt13_submissions.tgz
wget https://www.statmt.org/wmt13/wmt13-manual-evaluation.tgz -O data/wmt13_14/wmt13-manual-evaluation.tgz
echo "Unzipping Files .."
tar zxvf data/wmt13_14/wmt13_submissions.tgz -C data/wmt13_14
tar zxvf data/wmt13_14/wmt13-manual-evaluation.tgz -C data/wmt13_14
echo "Preprocessing files .."
mkdir -p data/wmt13_14/processed
python preprocess/prepare_data_wmt13_14.py --raw-dir ./data/wmt13_14 --processed-dir ./data/wmt13_14/processed --year 2013

echo "Merging WMT 2013 and 2014 data and splitting to train and val"
python scripts/utils/train_val_split.py --filenames data/wmt13_14/processed/wmt14-human-judgements.csv,data/wmt13_14/processed/wmt13-human-judgements.csv --train-output-file data/wmt13_14/processed/train.csv --val-output-file data/wmt13_14/processed/val.csv --val-frac 0.1
