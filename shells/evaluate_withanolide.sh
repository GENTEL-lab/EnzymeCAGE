python infer.py --config config/infer/case-study/withanolide/seed_40.yaml
python infer.py --config config/infer/case-study/withanolide/seed_41.yaml
python infer.py --config config/infer/case-study/withanolide/seed_42.yaml
python infer.py --config config/infer/case-study/withanolide/seed_43.yaml
python infer.py --config config/infer/case-study/withanolide/seed_44.yaml
cd scripts
python evaluate_external-test.py --pred_result_path ../checkpoints/pretrain/seed_42/data_epoch_19.csv --test_data_path ../dataset/case-study/withanolide/data.csv --data_type withanolide