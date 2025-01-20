cd dataset
# generate toy data
mkdir toy-data
head -n 200 unseen-enzymes/test.csv > toy-data/test.csv

# calculate feature
cd ../feature
python main.py --data_path ../dataset/toy-data/test.csv --pocket_dir ../dataset/pocket/alphafill_8A/ --skip_calc_mol_conformation

# run inference
cd ..
python infer.py --config config/demo/infer.yaml
