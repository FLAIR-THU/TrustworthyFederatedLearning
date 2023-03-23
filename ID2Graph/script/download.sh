wget -P data/avila http://archive.ics.uci.edu/ml/machine-learning-databases/00459/avila.zip
unzip data/avila/avila.zip -d data/avila
rm data/avila/avila.zip
mv data/avila/avila/avila-tr.txt data/avila/avila-tr.txt
mv data/avila/avila/avila-ts.txt data/avila/avila-ts.txt
rm -rf data/avila/avila
wget -P data/drive http://archive.ics.uci.edu/ml/machine-learning-databases/00325/Sensorless_drive_diagnosis.txt
wget -P data/nursery http://archive.ics.uci.edu/ml/machine-learning-databases/nursery/nursery.data
wget -P data/phishing "http://archive.ics.uci.edu/ml/machine-learning-databases/00327/Training Dataset.arff"
mv "data/phishing/Training Dataset.arff" "data/phishing/phishing.data"
sed -i '1,36d' data/phishing/phishing.data
