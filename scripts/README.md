## Data cleaning

First, download the compressed data from the RHEA website: [128.tar.bz2](https://ftp.expasy.org/databases/rhea/old%5Freleases/128.tar.bz2) (2023-07-12) and [137.tar.bz2](https://ftp.expasy.org/databases/rhea/old%5Freleases/137.tar.bz2) (2025-02-05), then unzip and rename the folders:

```shell
tar -xvf 128.tar.bz2
tar -xvf 137.tar.bz2
mv 128 2023-07-12
mv 137 2025-02-05
```

Then run the data cleaning script:
```shell
python rhea_data_cleaning.py --tsv_dir ./2023-07-12/tsv/ --sequence_path ../dataset/RHEA/2025-02-05/uid2seq.pkl --save_dir ./2023-07-12/processed/

python rhea_data_cleaning.py --tsv_dir ./2025-02-05/tsv/ --sequence_path ../dataset/RHEA/2025-02-05/uid2seq.pkl --save_dir ./2025-02-05/processed/
```







