# Extracting Conceptnet

```
cd data
wget https://s3.amazonaws.com/conceptnet/downloads/2018/edges/conceptnet-assertions-5.6.0.csv.gz
gzip -d conceptnet-assertions-5.6.0.csv.gz
cd ../scripts
python extract_conceptnet.py
python graph_construction.py
```