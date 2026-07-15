#!/usr/bin/env python3
"""Convert MMDet dumped predictions (.pkl) to COCO bbox result JSON."""

import argparse
import json
from pathlib import Path

import mmengine


def to_list(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def get_img_id(sample, fallback_index):
    if hasattr(sample, "img_id"):
        return int(sample.img_id)
    metainfo = getattr(sample, "metainfo", {}) or {}
    if "img_id" in metainfo:
        return int(metainfo["img_id"])
    if hasattr(sample, "get"):
        try:
            return int(sample.get("img_id", fallback_index))
        except Exception:
            pass
    return int(fallback_index)


def get_pred_instances(sample):
    if hasattr(sample, "pred_instances"):
        return sample.pred_instances
    if isinstance(sample, dict):
        return sample.get("pred_instances", sample.get("pred_instances_3d"))
    return None


def get_field(container, name):
    if isinstance(container, dict):
        if name not in container:
            raise KeyError(f"Prediction dict is missing key: {name}")
        return container[name]
    return getattr(container, name)


def load_cat_ids(ann_path):
    coco = json.loads(Path(ann_path).read_text(encoding="utf-8"))
    return [int(cat["id"]) for cat in sorted(coco["categories"], key=lambda c: int(c["id"]))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl", required=True, help="Path produced by mmdet_test.py --out.")
    parser.add_argument("--ann", required=True, help="Validation COCO annotation JSON.")
    parser.add_argument("--out", required=True, help="Output COCO prediction JSON.")
    parser.add_argument("--score-threshold", type=float, default=0.0)
    parser.add_argument("--max-per-image", type=int, default=300)
    args = parser.parse_args()

    cat_ids = load_cat_ids(args.ann)
    samples = mmengine.load(args.pkl)
    predictions = []

    for idx, sample in enumerate(samples):
        img_id = get_img_id(sample, idx)
        pred = get_pred_instances(sample)
        if pred is None:
            continue

        bboxes = to_list(get_field(pred, "bboxes"))
        scores = to_list(get_field(pred, "scores"))
        labels = to_list(get_field(pred, "labels"))

        rows = []
        for box, score, label in zip(bboxes, scores, labels):
            score = float(score)
            if score < args.score_threshold:
                continue
            label = int(label)
            if label < 0 or label >= len(cat_ids):
                continue
            x1, y1, x2, y2 = [float(v) for v in box]
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            if w <= 0 or h <= 0:
                continue
            rows.append({
                "image_id": img_id,
                "category_id": cat_ids[label],
                "bbox": [x1, y1, w, h],
                "score": score,
            })

        rows.sort(key=lambda item: item["score"], reverse=True)
        predictions.extend(rows[:args.max_per_image])

    Path(args.out).write_text(json.dumps(predictions, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(predictions)} predictions to {args.out}")


if __name__ == "__main__":
    main()
