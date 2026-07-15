# Copyright (c) OpenMMLab. All rights reserved.
"""Hooks for ODVG validation loss diagnostics."""

import copy
import json
import os.path as osp
from collections import defaultdict
from typing import Optional

import torch
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from mmengine.runner import Runner
from mmengine.runner.amp import autocast

from mmdet.registry import HOOKS


@HOOKS.register_module()
class ODVGValLossAndMetricHook(Hook):
    """Periodically evaluate ODVG validation loss and record metrics as JSONL.

    The normal validation loop still computes metrics such as COCO bbox mAP.
    This hook adds a loss-mode pass over an ODVG split and records both the
    loss pass and validation-loop metrics to one JSONL file.
    """

    priority = 'LOW'

    def __init__(self,
                 interval: int = 10000,
                 split: str = 'valid',
                 output: str = 'train_eval_records.jsonl',
                 data_root: Optional[str] = None,
                 batch_size: Optional[int] = None,
                 num_workers: int = 2,
                 max_batches: Optional[int] = None,
                 amp: bool = True) -> None:
        self.interval = interval
        self.split = split
        self.output = output
        self.data_root = data_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_batches = max_batches
        self.amp = amp
        self._dataloader = None
        self._output_path = None

    def before_train(self, runner: Runner) -> None:
        self._output_path = self.output
        if not osp.isabs(self._output_path):
            self._output_path = osp.join(runner.work_dir, self._output_path)
        self._dataloader = runner.build_dataloader(
            self._make_loss_dataloader_cfg(runner.cfg))

    def _make_loss_dataloader_cfg(self, cfg):
        dataloader_cfg = copy.deepcopy(cfg.train_dataloader)
        dataset_cfg = dataloader_cfg.dataset

        if self.data_root is not None:
            dataset_cfg.data_root = self.data_root
        dataset_cfg.ann_file = f'{self.split}/odvg_{self.split}.jsonl'
        dataset_cfg.data_prefix = dict(img=self.split)
        dataset_cfg.test_mode = False
        dataset_cfg.filter_cfg = dict(filter_empty_gt=False)

        if hasattr(cfg, 'train_pipeline'):
            pipeline = copy.deepcopy(cfg.train_pipeline)
            for index, transform in enumerate(pipeline):
                if transform.get('type') == 'RandomChoiceResize':
                    pipeline[index] = dict(
                        type='FixScaleResize',
                        scale=(800, 1333),
                        keep_ratio=True,
                        backend='pillow')
            dataset_cfg.pipeline = pipeline

        if self.batch_size is not None:
            dataloader_cfg.batch_size = self.batch_size
        dataloader_cfg.num_workers = self.num_workers
        dataloader_cfg.persistent_workers = bool(self.num_workers > 0)
        dataloader_cfg.sampler = dict(type='DefaultSampler', shuffle=False)
        dataloader_cfg.pop('batch_sampler', None)
        return dataloader_cfg

    def _write_record(self, record: dict) -> None:
        with open(self._output_path, 'a', encoding='utf-8') as file:
            file.write(json.dumps(record, ensure_ascii=False) + '\n')

    @staticmethod
    def _to_float(value):
        if isinstance(value, torch.Tensor):
            return float(value.detach().cpu())
        return float(value)

    def _evaluate_loss(self, runner: Runner) -> dict:
        model = (
            runner.model.module
            if is_model_wrapper(runner.model) else runner.model)

        # Keep every submodule (notably Dropout and normalization layers) in
        # eval mode, but enable the detector's training-only DINO branch. That
        # branch produces enc_outputs_class, enc_outputs_coord, and dn_meta,
        # which GroundingDINOHead.loss() requires.
        module_training_states = {
            module: module.training for module in runner.model.modules()
        }
        runner.model.eval()
        model.training = True

        totals = defaultdict(float)
        num_batches = 0
        num_samples = 0

        try:
            for batch_idx, data_batch in enumerate(
                    self._dataloader, start=1):
                if (self.max_batches is not None
                        and batch_idx > self.max_batches):
                    break
                with torch.no_grad():
                    with autocast(enabled=self.amp):
                        processed = model.data_preprocessor(
                            data_batch, training=True)
                        losses = model._run_forward(processed, mode='loss')
                        _, log_vars = model.parse_losses(losses)

                batch_size = len(processed.get('data_samples', [])) or 1
                num_batches += 1
                num_samples += batch_size
                for name, value in log_vars.items():
                    totals[name] += self._to_float(value) * batch_size
        finally:
            # Restore the exact state of every module instead of assuming the
            # whole model was uniformly in train mode before this hook ran.
            for module, training in module_training_states.items():
                module.training = training

        losses = {
            name: value / max(num_samples, 1)
            for name, value in sorted(totals.items())
        }
        return dict(
            type='val_loss',
            iter=runner.iter + 1,
            epoch=runner.epoch,
            split=self.split,
            num_batches=num_batches,
            num_samples=num_samples,
            losses=losses)

    def after_train_iter(self,
                         runner: Runner,
                         batch_idx: int,
                         data_batch: Optional[dict] = None,
                         outputs: Optional[dict] = None) -> None:
        if self.interval <= 0:
            return
        if not self.every_n_train_iters(runner, self.interval):
            return
        runner.logger.info(
            f'Evaluating {self.split} ODVG loss at iter {runner.iter + 1}')
        record = self._evaluate_loss(runner)
        self._write_record(record)
        if 'loss' in record['losses']:
            runner.logger.info(
                f"{self.split} ODVG loss: {record['losses']['loss']:.4f}")

    def after_val_epoch(self,
                        runner: Runner,
                        metrics: Optional[dict] = None) -> None:
        if metrics is None:
            return
        self._write_record(
            dict(
                type='val_metric',
                iter=runner.iter + 1,
                epoch=runner.epoch,
                metrics={
                    key: self._to_float(value)
                    for key, value in metrics.items()
                }))
