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

## Test dataset

We constructed two internal test sets, `Orphan-335` and `Enzyme-405`, for benchmarking on unseen reactions and unseen enzymes. The data for both test sets were sourced from the RHEA database, specifically from entries dated between 2023-07-12 and 2025-02-05. We began with 984 reactions, then removed any whose reaction similarity exceeded 0.9 to reactions in the training set, resulting in 630 unseen reactions. Among the 630 reactions, 295 reactions are catalyzed by enzymes that were not in the training set, these formed `Enzyme-405` test set of 405 enzymes and 295 reactions. And the remaining 335 reactions formed `Orphan-335` test set.

For the `Orphan-335` test set, each reaction is annotated in the database with enzymes proven to catalyze them and has at least one enzyme that appears in the training set. This test set was designed to evaluate the model’s ability to identify positive enzymes for orphan reactions within a specific enzyme pool.

The `Enzyme-405` test set comprises 961 enzyme-reaction pairs, encompassing 295 reactions and 405 enzymes, none of which appear in the 2023-07-12 RHEA snapshot. Unlike the `Orphan-335` test set, we constructed negative enzymes for each reaction in `Enzyme-405`. To do so, for each reaction in this test set, we identified the 10 most similar reactions from the 2023-07-12 RHEA snapshot and designated their corresponding enzymes as negative enzymes. This resulted in 14,960 negative pairs, which, combined with the 961 positives, produce a total of 15,921 enzyme–reaction pairs.




