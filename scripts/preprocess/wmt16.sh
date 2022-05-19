rm -rf data/wmt16
mkdir -p data/wmt16
echo "Downloading WMT 2016 Data.."
wget http://data.statmt.org/wmt16/translation-task/wmt16-submitted-data-v2.tgz -P data/wmt16
wget http://data.statmt.org/wmt16/translation-task/wmt16-translation-judgements.zip -P data/wmt16
echo "Unzipping Files .."
tar zxvf data/wmt16/wmt16-submitted-data-v2.tgz -C data/wmt16
unzip data/wmt16/wmt16-translation-judgements.zip -d data/wmt16
echo "Preprocessing files .."
mkdir -p data/wmt16/processed
python preprocess/prepare_data_wmt15_16.py --raw-dir ./data/wmt16 --processed-dir ./data/wmt16/processed --year 2016
