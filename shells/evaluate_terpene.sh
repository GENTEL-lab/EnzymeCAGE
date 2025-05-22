#python infer.py --config config/infer/external-test-set/terpene/wo-finetune.yaml
#python infer.py --config config/infer/external-test-set/terpene/with-finetune.yaml
cd scripts/
python evaluate_external-test.py --test_data_path ../dataset/external-test-set/terpene/test_Terpene.csv --pred_result_path ../checkpoints/pretrain/seed_42/test_Terpene_epoch_19.csv
python evaluate_external-test.py --test_data_path ../dataset/external-test-set/terpene/test_Terpene.csv --pred_result_path ../checkpoints/domain-specific-ft/terpene/seed_42/test_Terpene_epoch_9.csv
