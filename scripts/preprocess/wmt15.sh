rm -rf data/wmt16
mkdir -p data/wmt15
echo "Downloading WMT 2015 Data.."
wget https://www.statmt.org/wmt15/wmt15-submitted-data.tgz -P data/wmt15
wget https://www.statmt.org/wmt15/translation-judgements.zip -P data/wmt15
echo "Unzipping Files .."
tar zxvf data/wmt15/wmt15-submitted-data.tgz -C data/wmt15
unzip data/wmt15/translation-judgements.zip -d data/wmt15
echo "Preprocessing files .."
mkdir -p data/wmt15/processed
python preprocess/prepare_data_wmt15_16.py --raw-dir ./data/wmt15 --processed-dir ./data/wmt15/processed --year 2015
