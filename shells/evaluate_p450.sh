python infer.py --config config/infer/external-test-set/p450/wo-finetune.yaml
python infer.py --config config/infer/external-test-set/p450/with-finetune.yaml
cd scripts/
python evaluate_external-test.py --test_data_path ../dataset/external-test-set/p450/test_P450.csv --pred_result_path ../checkpoints/pretrain/seed_42/test_P450_epoch_19.csv
python evaluate_external-test.py --test_data_path ../dataset/external-test-set/p450/test_P450.csv --pred_result_path ../checkpoints/domain-specific-ft/p450/seed_42/test_P450_epoch_9.csv