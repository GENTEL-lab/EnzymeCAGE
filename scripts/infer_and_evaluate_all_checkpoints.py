import argparse
import json
import os
import sys
from datetime import datetime
from types import SimpleNamespace

import yaml

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from evaluate import evaluate_result
from infer import inference
from utils import check_dir, check_files, seed_everything


METRIC_KEYS = [
    "top10_dcg",
    "top1_percent_ef",
    "top2_percent_ef",
    "top1_sr",
    "top3_sr",
    "top5_sr",
    "top10_sr",
]


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def list_checkpoint_files(ckpt_dir):
    names = []
    for name in os.listdir(ckpt_dir):
        if name == "best_model.pth":
            names.append(name)
        elif name.startswith("epoch_") and name.endswith(".pth"):
            names.append(name)

    def sort_key(name):
        if name == "best_model.pth":
            return (1, 10**9)
        epoch = int(name.removeprefix("epoch_").removesuffix(".pth"))
        return (0, epoch)

    return sorted(names, key=sort_key)


def model_label(model_name):
    return model_name.removesuffix(".pth")


def build_model_conf(base_conf, ckpt_dir, model_list):
    conf = dict(base_conf)
    conf["ckpt_dir"] = ckpt_dir
    conf["model_list"] = model_list
    return SimpleNamespace(**conf)


def evaluate_all_predictions(ckpt_dir, dataset_stem):
    metrics_by_model = {}
    for csv_name in sorted(os.listdir(ckpt_dir)):
        if not csv_name.startswith(f"{dataset_stem}_") or not csv_name.endswith(".csv"):
            continue

        result_path = os.path.join(ckpt_dir, csv_name)
        ckpt_label = csv_name.removeprefix(f"{dataset_stem}_").removesuffix(".csv")
        metrics = evaluate_result(result_path=result_path, pos_pair_db_path=result_path)
        metrics_by_model[ckpt_label] = metrics

        metrics_path = os.path.join(ckpt_dir, f"{dataset_stem}_{ckpt_label}_metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
            f.write("\n")

    return metrics_by_model


def build_summary(dataset_stem, ckpt_dir, metrics_by_model):
    best_by_metric = {}
    for metric in METRIC_KEYS:
        best_model, best_metrics = max(
            metrics_by_model.items(),
            key=lambda item: item[1][metric],
        )
        best_by_metric[metric] = {
            "model": best_model,
            "value": best_metrics[metric],
        }

    return {
        "dataset": dataset_stem,
        "checkpoint_dir": ckpt_dir,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "models": metrics_by_model,
        "best_by_metric": best_by_metric,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Base infer config path.")
    parser.add_argument("--ckpt-dir", required=True, help="Checkpoint directory to evaluate.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Optional checkpoint file names to run, for example: epoch_0.pth best_model.pth",
    )
    parser.add_argument(
        "--summary-path",
        default=None,
        help="Optional summary json path. Defaults to <ckpt-dir>/<dataset>_metrics_summary.json.",
    )
    parser.add_argument(
        "--evaluate-existing-only",
        action="store_true",
        help="Skip inference and only evaluate existing prediction csv files in the checkpoint directory.",
    )
    args = parser.parse_args()

    ckpt_dir = os.path.abspath(args.ckpt_dir)
    assert os.path.isdir(ckpt_dir), f"Checkpoint dir not found: {ckpt_dir}"

    base_conf = load_config(args.config)
    model_list = list_checkpoint_files(ckpt_dir)
    assert model_list, f"No checkpoint files found in: {ckpt_dir}"
    if args.models:
        allowed = set(args.models)
        model_list = [name for name in model_list if name in allowed]
        assert model_list, "No requested checkpoints matched files in the checkpoint directory."

    model_conf = build_model_conf(base_conf, ckpt_dir, model_list)
    if not args.evaluate_existing_only:
        seed = 42 if not hasattr(model_conf, "seed") else model_conf.seed
        seed_everything(seed)
        check_files(model_conf)
        inference(model_conf)

    dataset_stem = os.path.splitext(os.path.basename(model_conf.data_path))[0]
    metrics_by_model = evaluate_all_predictions(ckpt_dir, dataset_stem)
    summary = build_summary(dataset_stem, ckpt_dir, metrics_by_model)

    summary_path = args.summary_path
    if not summary_path:
        summary_path = os.path.join(ckpt_dir, f"{dataset_stem}_metrics_summary.json")
    summary_path = os.path.abspath(summary_path)
    check_dir(summary_path)

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")

    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
