python retrieve.py --data_path dataset/case-study/glutarate/rxns.csv --exclude_rxns dataset/case-study/glutarate/rxns.csv --db_path ./dataset/training/train.csv
python infer.py --config config/infer/case-study/glutarate.yaml
python scripts/evaluate_glutarate.py
