#!/usr/bin/env python3
"""Evaluate ODVG validation loss for LLMDet/MMDetection configs.

This script is intended for detection-only fine-tuning diagnostics. It reuses
the training pipeline, swaps the ODVG annotation split to validation data, and
runs the model in loss mode without gradients.
"""

import argparse
import copy
import json
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import torch
from mmengine.config import Config, DictAction
from mmengine.runner import Runner
from mmengine.runner.amp import autocast
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ODVG validation loss.")
    parser.add_argument("config", help="MMDetection/LLMDet config file.")
    parser.add_argument("checkpoints", nargs="+", help="Checkpoint path(s).")
    parser.add_argument(
        "--split",
        default="valid",
        choices=["train", "valid"],
        help="ODVG split to evaluate. Defaults to valid.",
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help="Override dataset data_root. Defaults to config value.",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--amp", action="store_true", help="Use AMP during loss eval.")
    parser.add_argument("--output", default=None, help="Optional JSON output path.")
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="Override config options, same format as mmdet_train.py.",
    )
    return parser.parse_args()


def to_float(value):
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu())
    return float(value)


def make_loss_dataloader_cfg(cfg, args):
    dataloader_cfg = copy.deepcopy(cfg.train_dataloader)
    dataset_cfg = dataloader_cfg.dataset

    if args.data_root is not None:
        dataset_cfg.data_root = args.data_root
    dataset_cfg.ann_file = f"{args.split}/odvg_{args.split}.jsonl"
    dataset_cfg.data_prefix = dict(img=args.split)
    dataset_cfg.test_mode = False
    dataset_cfg.filter_cfg = dict(filter_empty_gt=False)

    if hasattr(cfg, "train_pipeline"):
        pipeline = copy.deepcopy(cfg.train_pipeline)
        for index, transform in enumerate(pipeline):
            if transform.get("type") == "RandomChoiceResize":
                pipeline[index] = dict(
                    type="FixScaleResize",
                    scale=(800, 1333),
                    keep_ratio=True,
                    backend="pillow",
                )
        dataset_cfg.pipeline = pipeline

    if args.batch_size is not None:
        dataloader_cfg.batch_size = args.batch_size
    dataloader_cfg.num_workers = args.num_workers
    dataloader_cfg.persistent_workers = bool(args.num_workers > 0)
    dataloader_cfg.sampler = dict(type="DefaultSampler", shuffle=False)
    dataloader_cfg.pop("batch_sampler", None)
    return dataloader_cfg


def build_runner(config_path, args):
    cfg = Config.fromfile(config_path)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    cfg.launcher = "none"
    cfg.work_dir = tempfile.mkdtemp(prefix="llmdet_val_loss_")
    cfg.load_from = None
    return Runner.from_cfg(cfg), cfg


def evaluate_checkpoint(runner, cfg, checkpoint, args):
    runner.load_checkpoint(checkpoint, map_location="cpu")
    model = runner.model
    model.eval()

    dataloader_cfg = make_loss_dataloader_cfg(cfg, args)
    dataloader = runner.build_dataloader(dataloader_cfg)

    totals = defaultdict(float)
    num_batches = 0
    num_samples = 0

    progress = tqdm(dataloader, desc=f"loss {Path(checkpoint).name}")
    for batch_index, data_batch in enumerate(progress, start=1):
        if args.max_batches is not None and batch_index > args.max_batches:
            break
        with torch.no_grad():
            with autocast(enabled=args.amp):
                processed = model.data_preprocessor(data_batch, training=True)
                losses = model._run_forward(processed, mode="loss")
                _, log_vars = model.parse_losses(losses)

        batch_size = len(processed.get("data_samples", [])) or 1
        num_batches += 1
        num_samples += batch_size
        for name, value in log_vars.items():
            totals[name] += to_float(value) * batch_size

        if "loss" in log_vars:
            progress.set_postfix(loss=f"{to_float(log_vars['loss']):.4f}")

    averages = {
        name: value / max(num_samples, 1)
        for name, value in sorted(totals.items())
    }
    return {
        "checkpoint": str(Path(checkpoint).expanduser().resolve()),
        "split": args.split,
        "num_batches": num_batches,
        "num_samples": num_samples,
        "losses": averages,
    }


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    runner, cfg = build_runner(args.config, args)
    results = [
        evaluate_checkpoint(runner, cfg, checkpoint, args)
        for checkpoint in args.checkpoints
    ]

    print(json.dumps(results, indent=2))
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        print(f"Saved validation loss JSON to {output_path}")


if __name__ == "__main__":
    main()
