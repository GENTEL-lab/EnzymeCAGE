python infer.py --config config/infer/external-test-set/phosphatase/wo-finetune.yaml
python infer.py --config config/infer/external-test-set/phosphatase/with-finetune.yaml
cd scripts/
python evaluate_external-test.py --test_data_path ../dataset/external-test-set/phosphatase/test_Phosphatase.csv --pred_result_path ../checkpoints/pretrain/seed_42/test_Phosphatase_epoch_19.csv
python evaluate_external-test.py --test_data_path ../dataset/external-test-set/phosphatase/test_Phosphatase.csv --pred_result_path ../checkpoints/domain-specific-ft/phosphatase/seed_42/test_Phosphatase_epoch_9.csv