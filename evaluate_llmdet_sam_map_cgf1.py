#!/usr/bin/env python3
"""Standalone LLMDet + SAM mask evaluation for COCO data.

Pipeline:
    COCO image -> MMDet-compatible multi-class LLMDet inference
    -> SAM box-prompted masks -> COCO APmask + closed-set cgF1.

The default ``mmdet-standard`` mode submits the full ordered category list to
LLMDet once per image. It preserves LLMDet's native labels, scores, and
``test_cfg.max_per_img`` output, so its bbox AP is directly comparable with
``mmdet_test.py``. The cgF1 implementation is embedded here so this file does
not import SAM3_LoRA_1 or any other project evaluation script.

Required runtime packages:
    torch, numpy, pillow, pycocotools, segment-anything
    plus the local LLMDet repository and its MMDetection dependencies.
"""

import argparse
import contextlib
import json
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **_kwargs):
        return iterable


COCO_BBOX_METRICS = [
    "APbbox", "APbbox50", "APbbox75", "APbbox_small", "APbbox_medium",
    "APbbox_large", "ARbbox_maxDets1", "ARbbox_maxDets10",
    "ARbbox_maxDets100", "ARbbox_small", "ARbbox_medium", "ARbbox_large",
]
COCO_MASK_METRICS = [
    "APmask", "APmask50", "APmask75", "APmask_small", "APmask_medium",
    "APmask_large", "ARmask_maxDets1", "ARmask_maxDets10",
    "ARmask_maxDets100", "ARmask_small", "ARmask_medium", "ARmask_large",
]
CGF1_IOU_THRESHOLDS = [round(0.5 + 0.05 * index, 2) for index in range(10)]


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def dump_json(path, payload):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file)


def resolve_existing_path(value, option_name):
    path = Path(value).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{option_name} does not exist: {path}")
    return path


def resolve_gt_path(args):
    if args.gt_json:
        return Path(args.gt_json)
    if args.val_data_dir:
        return Path(args.val_data_dir) / "_annotations.coco.json"
    raise ValueError("Either --gt-json or --val-data-dir is required.")


def resolve_image_root(args, gt_path):
    if args.image_root:
        return Path(args.image_root)
    if args.val_data_dir:
        return Path(args.val_data_dir)
    return Path(gt_path).parent


def normalize_name(value):
    text = str(value).strip().lower()
    for character in ".,:;()/\\|[]{}\"'`":
        text = text.replace(character, " ")
    return " ".join(text.split())


def build_category_maps(gt_data):
    id_to_name = {}
    name_to_id = {}
    for category in gt_data.get("categories", []):
        if "id" not in category or "name" not in category:
            continue
        category_id = int(category["id"])
        category_name = str(category["name"]).strip()
        if category_name:
            id_to_name[category_id] = category_name
            name_to_id[normalize_name(category_name)] = category_id
    return id_to_name, name_to_id


def build_image_present_category_ids(gt_data):
    valid_categories = {int(category["id"]) for category in gt_data.get("categories", [])}
    present = defaultdict(set)
    for annotation in gt_data.get("annotations", []):
        category_id = int(annotation.get("category_id", -1))
        if category_id in valid_categories:
            present[int(annotation["image_id"])].add(category_id)
    return {image_id: sorted(category_ids) for image_id, category_ids in present.items()}


def prepare_gt_for_eval(gt_data):
    prepared = dict(gt_data)
    prepared["images"] = [dict(image) for image in gt_data.get("images", [])]
    prepared["annotations"] = [dict(annotation) for annotation in gt_data.get("annotations", [])]
    prepared["categories"] = [dict(category) for category in gt_data.get("categories", [])]
    for image in prepared["images"]:
        image.setdefault("is_instance_exhaustive", True)
    for index, annotation in enumerate(prepared["annotations"], start=1):
        annotation.setdefault("id", index)
        annotation.setdefault("iscrowd", 0)
        annotation.setdefault("ignore", 0)
        if "area" not in annotation and "bbox" in annotation:
            annotation["area"] = float(annotation["bbox"][2]) * float(annotation["bbox"][3])
    return prepared


def filter_gt_to_categories(gt_data, category_ids):
    category_ids = {int(category_id) for category_id in category_ids}
    filtered = dict(gt_data)
    filtered["images"] = [dict(image) for image in gt_data.get("images", [])]
    filtered["categories"] = [
        dict(category)
        for category in gt_data.get("categories", [])
        if int(category["id"]) in category_ids
    ]
    filtered["annotations"] = [
        dict(annotation)
        for annotation in gt_data.get("annotations", [])
        if int(annotation["category_id"]) in category_ids
    ]
    if not filtered["categories"]:
        raise ValueError(f"No categories remain after filtering: {sorted(category_ids)}")
    return filtered


def resolve_eval_category_ids(args, gt_data):
    category_ids = set(args.eval_category_id or [])
    if args.eval_category_name:
        _, name_to_id = build_category_maps(gt_data)
        for category_name in args.eval_category_name:
            normalized = normalize_name(category_name)
            if normalized not in name_to_id:
                raise ValueError(f"Unknown --eval-category-name: {category_name}")
            category_ids.add(name_to_id[normalized])
    return sorted(category_ids)


def limit_gt_to_images(gt_data, max_images):
    if max_images is None:
        return gt_data
    images = [dict(image) for image in gt_data.get("images", [])[:max_images]]
    image_ids = {int(image["id"]) for image in images}
    return {
        **gt_data,
        "images": images,
        "annotations": [
            dict(annotation)
            for annotation in gt_data.get("annotations", [])
            if int(annotation["image_id"]) in image_ids
        ],
        "categories": [dict(category) for category in gt_data.get("categories", [])],
    }


def has_gt_segmentation(gt_data):
    return any(annotation.get("segmentation") for annotation in gt_data.get("annotations", []))


def clip_xyxy(box, width, height):
    x1, y1, x2, y2 = [float(value) for value in box]
    return [
        min(max(x1, 0.0), float(width)),
        min(max(y1, 0.0), float(height)),
        min(max(x2, 0.0), float(width)),
        min(max(y2, 0.0), float(height)),
    ]


def xyxy_to_xywh(box):
    x1, y1, x2, y2 = [float(value) for value in box]
    return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]


def bbox_iou_xyxy(first, second):
    ax1, ay1, ax2, ay2 = [float(value) for value in first]
    bx1, by1, bx2, by2 = [float(value) for value in second]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if intersection <= 0.0:
        return 0.0
    area_first = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_second = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denominator = area_first + area_second - intersection
    return intersection / denominator if denominator > 0.0 else 0.0


def run_nms(boxes, scores, iou_threshold):
    if iou_threshold is None or len(boxes) <= 1:
        return list(range(len(boxes)))
    order = sorted(range(len(scores)), key=lambda index: float(scores[index]), reverse=True)
    keep = []
    while order:
        current = order.pop(0)
        keep.append(current)
        order = [
            index for index in order
            if bbox_iou_xyxy(boxes[current], boxes[index]) <= float(iou_threshold)
        ]
    return keep


def rle_to_jsonable(rle):
    rle = dict(rle)
    if isinstance(rle.get("counts"), bytes):
        rle["counts"] = rle["counts"].decode("ascii")
    return rle


def encode_binary_mask(mask):
    return rle_to_jsonable(mask_utils.encode(np.asfortranarray(mask.astype(np.uint8))))


def load_sam(args, device):
    try:
        from segment_anything import SamPredictor, sam_model_registry
    except ImportError as exc:
        raise RuntimeError("Install the segment-anything package to run SAM segmentation.") from exc
    checkpoint = resolve_existing_path(args.sam_checkpoint, "--sam-checkpoint")
    sam = sam_model_registry[args.sam_model_type](checkpoint=str(checkpoint))
    sam.to(device)
    sam.eval()
    return SamPredictor(sam), checkpoint


def load_llmdet(args, device):
    repo_path = resolve_existing_path(args.llmdet_repo, "--llmdet-repo")
    config_path = resolve_existing_path(args.llmdet_config, "--llmdet-config")
    checkpoint_path = resolve_existing_path(args.llmdet_checkpoint, "--llmdet-checkpoint")
    if not repo_path.is_dir():
        raise NotADirectoryError(f"--llmdet-repo must be a directory: {repo_path}")

    repo_text = str(repo_path)
    if repo_text not in sys.path:
        sys.path.insert(0, repo_text)
    try:
        from mmdet.apis import DetInferencer
    except ImportError as exc:
        raise RuntimeError(
            "Could not import LLMDet's DetInferencer. Use the Python environment "
            "where LLMDet and its MMDetection dependencies are installed."
        ) from exc

    inferencer = DetInferencer(
        model=str(config_path), weights=str(checkpoint_path), device=device
    )
    try:
        inferencer.model.test_cfg.chunked_size = int(args.llmdet_chunked_size)
    except (AttributeError, TypeError):
        pass
    return inferencer, repo_path, config_path, checkpoint_path


def build_llmdet_queries(image_id, args, gt_data, id_to_name, present_category_ids):
    if args.query_mode == "present-categories":
        category_ids = present_category_ids.get(int(image_id), [])
    else:
        category_ids = sorted(int(category["id"]) for category in gt_data.get("categories", []))
    return [
        (category_id, id_to_name[category_id], f"{id_to_name[category_id]} .")
        for category_id in category_ids
        if category_id in id_to_name
    ]


def build_mmdet_standard_query(id_to_name):
    """Return the dataset text list and its LLMDet-label to COCO-id mapping."""
    category_ids = sorted(id_to_name)
    category_names = [id_to_name[category_id] for category_id in category_ids]
    return category_names, dict(enumerate(category_ids))


def predict_llmdet_boxes(inferencer, image_path, prompt, args):
    result = inferencer(
        str(image_path),
        texts=[prompt],
        custom_entities=True,
        pred_score_thr=float(args.box_threshold),
        no_save_vis=True,
        no_save_pred=True,
        return_vis=False,
        batch_size=1,
    )
    predictions = result.get("predictions", [])
    if len(predictions) != 1:
        raise RuntimeError(
            f"LLMDet returned {len(predictions)} prediction records for: {image_path}"
        )
    prediction = predictions[0]
    boxes = prediction.get("bboxes", [])
    scores = prediction.get("scores", [])
    labels = prediction.get("labels", list(range(len(boxes))))
    if len(labels) != len(boxes):
        labels = list(range(len(boxes)))
    return boxes, scores, labels


def generate_llmdet_sam_predictions(gt_data, gt_path, args):
    try:
        import torch
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("LLMDet + SAM inference requires torch and Pillow.") from exc

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    image_root = resolve_image_root(args, gt_path)
    id_to_name, _ = build_category_maps(gt_data)
    present_category_ids = build_image_present_category_ids(gt_data)
    images = [dict(image) for image in gt_data.get("images", [])]
    standard_texts, standard_label_to_category_id = build_mmdet_standard_query(id_to_name)

    print("\n" + "=" * 80)
    print("LLMDET + SAM MASK INFERENCE")
    print("=" * 80)
    print(f"Images: {len(images)}")
    print(f"Query mode: {args.query_mode}")
    print(f"Box threshold: {args.box_threshold:.3f}")
    if args.query_mode == "mmdet-standard":
        print("Post-processing: native LLMDet test_cfg output (no external NMS or Top-K)")
        print(f"Classes in one query: {len(standard_texts)}")
    else:
        print(f"NMS IoU: {args.nms_iou}" if args.nms_iou is not None else "NMS: disabled")
        print(f"Max detections per query: {args.max_detections_per_query}")
    print(f"SAM box batch size: {args.sam_box_batch_size}")

    inferencer, repo_path, config_path, checkpoint_path = load_llmdet(args, device)
    sam_predictor, sam_checkpoint = load_sam(args, device)
    args.llmdet_repo = str(repo_path)
    args.llmdet_config = str(config_path)
    args.llmdet_checkpoint = str(checkpoint_path)
    args.sam_checkpoint = str(sam_checkpoint)

    predictions = []
    skipped_missing_images = 0
    skipped_empty_boxes = 0
    for index, image_info in enumerate(tqdm(images, desc="LLMDet+SAM"), start=1):
        image_path = image_root / str(image_info["file_name"])
        if not image_path.exists():
            skipped_missing_images += 1
            if args.strict:
                raise FileNotFoundError(f"Image not found: {image_path}")
            continue

        pil_image = Image.open(image_path).convert("RGB")
        width, height = pil_image.size
        np_image = np.array(pil_image)
        detections = []
        if args.query_mode == "mmdet-standard":
            boxes, scores, labels = predict_llmdet_boxes(
                inferencer, image_path, standard_texts, args
            )
            for box, score, label in zip(boxes, scores, labels):
                label = int(label)
                if label not in standard_label_to_category_id:
                    raise RuntimeError(
                        "LLMDet returned a label outside the configured class list: "
                        f"{label} not in [0, {len(standard_texts) - 1}]."
                    )
                score = float(score)
                if score < float(args.box_threshold):
                    continue
                box = clip_xyxy(box, width, height)
                if xyxy_to_xywh(box)[2] <= 0.0 or xyxy_to_xywh(box)[3] <= 0.0:
                    skipped_empty_boxes += 1
                    continue
                category_id = standard_label_to_category_id[label]
                detections.append(
                    {
                        "box_xyxy": box,
                        "score": score,
                        "category_id": category_id,
                        "category_name": id_to_name[category_id],
                        "text_label": str(label),
                        "prompt": list(standard_texts),
                    }
                )
        else:
            queries = build_llmdet_queries(
                image_info["id"], args, gt_data, id_to_name, present_category_ids
            )
            for category_id, category_name, prompt in queries:
                boxes, scores, labels = predict_llmdet_boxes(inferencer, image_path, prompt, args)
                valid_boxes, valid_scores, valid_labels = [], [], []
                for box, score, label in zip(boxes, scores, labels):
                    score = float(score)
                    if score < float(args.box_threshold):
                        continue
                    box = clip_xyxy(box, width, height)
                    if xyxy_to_xywh(box)[2] <= 0.0 or xyxy_to_xywh(box)[3] <= 0.0:
                        skipped_empty_boxes += 1
                        continue
                    valid_boxes.append(box)
                    valid_scores.append(score)
                    valid_labels.append(label)

                keep = run_nms(valid_boxes, valid_scores, args.nms_iou)
                if args.max_detections_per_query is not None:
                    keep = keep[: args.max_detections_per_query]
                for keep_index in keep:
                    detections.append(
                        {
                            "box_xyxy": valid_boxes[keep_index],
                            "score": valid_scores[keep_index],
                            "category_id": int(category_id),
                            "category_name": category_name,
                            "text_label": str(valid_labels[keep_index]),
                            "prompt": prompt,
                        }
                    )

        if detections:
            sam_predictor.set_image(np_image)
            for start in range(0, len(detections), args.sam_box_batch_size):
                detection_batch = detections[start:start + args.sam_box_batch_size]
                input_boxes = torch.tensor(
                    [detection["box_xyxy"] for detection in detection_batch],
                    dtype=torch.float32,
                    device=device,
                )
                with torch.no_grad():
                    transformed_boxes = sam_predictor.transform.apply_boxes_torch(
                        input_boxes, np_image.shape[:2]
                    )
                    masks, sam_scores, _ = sam_predictor.predict_torch(
                        point_coords=None,
                        point_labels=None,
                        boxes=transformed_boxes,
                        multimask_output=False,
                    )

                masks_np = masks[:, 0].detach().cpu().numpy().astype(np.uint8)
                sam_scores_np = (
                    sam_scores[:, 0].detach().cpu().numpy().tolist()
                    if sam_scores is not None
                    else [1.0] * len(detection_batch)
                )
                for detection, mask, sam_score in zip(detection_batch, masks_np, sam_scores_np):
                    rle = encode_binary_mask(mask)
                    det_score = float(detection["score"])
                    final_score = det_score * float(sam_score) if args.use_sam_score else det_score
                    predictions.append(
                        {
                            "image_id": int(image_info["id"]),
                            "category_id": int(detection["category_id"]),
                            "bbox": [float(value) for value in xyxy_to_xywh(detection["box_xyxy"])],
                            "segmentation": rle,
                            "area": float(mask_utils.area(rle)),
                            "score": float(final_score),
                            "det_score": det_score,
                            "sam_score": float(sam_score),
                            "category_name": detection["category_name"],
                            "text_label": detection["text_label"],
                            "prompt": detection["prompt"],
                            "file_name": image_info["file_name"],
                        }
                    )

        if args.log_every > 0 and (
            index == 1 or index % args.log_every == 0 or index == len(images)
        ):
            print(f"[{index}/{len(images)}] mask predictions so far: {len(predictions)}")

    if skipped_missing_images:
        print(f"Skipped missing images: {skipped_missing_images}")
    if skipped_empty_boxes:
        print(f"Skipped empty LLMDet boxes: {skipped_empty_boxes}")
    return predictions


def run_coco_eval(gt_file, pred_file, iou_type):
    metric_names = COCO_BBOX_METRICS if iou_type == "bbox" else COCO_MASK_METRICS
    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull):
            coco_gt = COCO(str(gt_file))
            coco_dt = coco_gt.loadRes(str(pred_file))
            coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
            coco_eval.params.useCats = True
            coco_eval.evaluate()
            coco_eval.accumulate()
    coco_eval.summarize()
    return {name: float(value) for name, value in zip(metric_names, coco_eval.stats)}


def bbox_iou_xywh(detection_box, gt_box, gt_iscrowd=False):
    dx1, dy1, dw, dh = [float(value) for value in detection_box]
    gx1, gy1, gw, gh = [float(value) for value in gt_box]
    dx2, dy2 = dx1 + dw, dy1 + dh
    gx2, gy2 = gx1 + gw, gy1 + gh
    ix1, iy1 = max(dx1, gx1), max(dy1, gy1)
    ix2, iy2 = min(dx2, gx2), min(dy2, gy2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if intersection <= 0.0:
        return 0.0
    detection_area = max(0.0, dw) * max(0.0, dh)
    gt_area = max(0.0, gw) * max(0.0, gh)
    denominator = detection_area if gt_iscrowd else detection_area + gt_area - intersection
    return min(1.0, intersection / denominator) if denominator > 0.0 else 0.0


def segmentation_to_rle(segmentation, height, width):
    if not segmentation:
        return None
    if isinstance(segmentation, dict):
        rle = dict(segmentation)
        if isinstance(rle.get("counts"), str):
            rle["counts"] = rle["counts"].encode("ascii")
        return rle
    if isinstance(segmentation, list):
        rles = mask_utils.frPyObjects(segmentation, int(height), int(width))
        return mask_utils.merge(rles) if isinstance(rles, list) else rles
    return None


def mask_iou_rle(detection_rle, gt_rle, gt_iscrowd=False):
    if detection_rle is None or gt_rle is None:
        return 0.0
    try:
        return float(mask_utils.iou([detection_rle], [gt_rle], [int(bool(gt_iscrowd))])[0, 0])
    except Exception:
        return 0.0


def hungarian_minimize(costs):
    rows = len(costs)
    columns = len(costs[0]) if rows else 0
    if rows == 0 or columns == 0:
        return []
    if rows > columns:
        raise ValueError("hungarian_minimize requires rows <= columns")
    u = [0.0] * (rows + 1)
    v = [0.0] * (columns + 1)
    p = [0] * (columns + 1)
    way = [0] * (columns + 1)
    for row in range(1, rows + 1):
        p[0] = row
        column_zero = 0
        minimum = [float("inf")] * (columns + 1)
        used = [False] * (columns + 1)
        while True:
            used[column_zero] = True
            row_zero = p[column_zero]
            delta = float("inf")
            next_column = 0
            for column in range(1, columns + 1):
                if used[column]:
                    continue
                current = costs[row_zero - 1][column - 1] - u[row_zero] - v[column]
                if current < minimum[column]:
                    minimum[column] = current
                    way[column] = column_zero
                if minimum[column] < delta:
                    delta = minimum[column]
                    next_column = column
            for column in range(columns + 1):
                if used[column]:
                    u[p[column]] += delta
                    v[column] -= delta
                else:
                    minimum[column] -= delta
            column_zero = next_column
            if p[column_zero] == 0:
                break
        while True:
            previous_column = way[column_zero]
            p[column_zero] = p[previous_column]
            column_zero = previous_column
            if column_zero == 0:
                break
    return [(p[column] - 1, column - 1) for column in range(1, columns + 1) if p[column]]


def max_weight_assignment(weights):
    rows = len(weights)
    columns = len(weights[0]) if rows else 0
    if rows == 0 or columns == 0:
        return []
    if rows <= columns:
        return hungarian_minimize([[1.0 - float(value) for value in row] for row in weights])
    transposed = [
        [1.0 - float(weights[row][column]) for row in range(rows)]
        for column in range(columns)
    ]
    matches = hungarian_minimize(transposed)
    return [(detection_index, gt_index) for gt_index, detection_index in matches]


def f1_from_counts(true_positive, false_positive, false_negative):
    precision = true_positive / (true_positive + false_positive + 1e-4)
    recall = true_positive / (true_positive + false_negative + 1e-4)
    return 2.0 * precision * recall / (precision + recall + 1e-4)


def evaluate_cgf1_query(gt_annotations, detections, score_threshold, iou_type, image_info):
    """One image-category query, matching SAM3_LoRA_1 CGF1Evaluator(use_cats=True)."""
    gt = [annotation for annotation in gt_annotations if not bool(annotation.get("ignore", 0))]
    detections = [
        detection for detection in detections
        if float(detection.get("score", 0.0)) >= score_threshold
    ]
    result = {"IL_TP": 0, "IL_TN": 0, "IL_FP": 0, "IL_FN": 0, "num_dt": len(detections)}
    if not gt and not detections:
        result["IL_TN"] = 1
        return result
    if gt and not detections:
        result.update(
            {
                "IL_FN": 1,
                "TPs": [0] * len(CGF1_IOU_THRESHOLDS),
                "FPs": [0] * len(CGF1_IOU_THRESHOLDS),
                "FNs": [len(gt)] * len(CGF1_IOU_THRESHOLDS),
                "local_positive_F1s": [0.0] * len(CGF1_IOU_THRESHOLDS),
            }
        )
        return result

    if iou_type == "bbox":
        ious = [
            [bbox_iou_xywh(detection["bbox"], annotation["bbox"], bool(annotation.get("iscrowd", 0))) for annotation in gt]
            for detection in detections
        ]
    elif iou_type == "segm":
        height, width = int(image_info["height"]), int(image_info["width"])
        gt_rles = [segmentation_to_rle(annotation.get("segmentation"), height, width) for annotation in gt]
        detection_rles = [
            segmentation_to_rle(detection.get("segmentation"), height, width)
            for detection in detections
        ]
        ious = [
            [mask_iou_rle(detection_rle, gt_rle, bool(annotation.get("iscrowd", 0))) for gt_rle, annotation in zip(gt_rles, gt)]
            for detection_rle in detection_rles
        ]
    else:
        raise ValueError(f"Unsupported cgF1 IoU type: {iou_type}")

    matches = max_weight_assignment(ious)
    match_scores = [ious[detection_index][gt_index] for detection_index, gt_index in matches]
    true_positives, false_positives, false_negatives, local_f1s = [], [], [], []
    for threshold in CGF1_IOU_THRESHOLDS:
        true_positive = sum(score >= threshold for score in match_scores)
        false_positive = len(detections) - true_positive
        false_negative = len(gt) - true_positive
        true_positives.append(true_positive)
        false_positives.append(false_positive)
        false_negatives.append(false_negative)
        local_f1s.append(f1_from_counts(true_positive, false_positive, false_negative))
    result.update(
        {
            "IL_TP": int(bool(gt) and bool(detections)),
            "IL_FP": int(not gt and bool(detections)),
            "IL_TN": int(not gt and not detections),
            "IL_FN": int(bool(gt) and not detections),
            "TPs": true_positives,
            "FPs": false_positives,
            "FNs": false_negatives,
        }
    )
    if gt and detections:
        result["local_positive_F1s"] = local_f1s
    return result


def mean(values):
    return sum(values) / max(len(values), 1)


def run_sam3_lora_compatible_cgf1(gt_data, predictions, iou_type, score_threshold=0.5):
    """Closed-set cgF1 used by SAM3_LoRA_1: every image x category pair is scored."""
    images_by_id = {int(image["id"]): image for image in gt_data.get("images", [])}
    image_ids = [
        int(image["id"]) for image in gt_data.get("images", [])
        if bool(image.get("is_instance_exhaustive", True))
    ]
    category_ids = sorted(int(category["id"]) for category in gt_data.get("categories", [])) or [1]
    gt_by_pair = defaultdict(list)
    for annotation in gt_data.get("annotations", []):
        if int(annotation["image_id"]) in images_by_id:
            gt_by_pair[(int(annotation["image_id"]), int(annotation.get("category_id", -1)))].append(annotation)
    detections_by_pair = defaultdict(list)
    for detection in predictions:
        if int(detection["image_id"]) in images_by_id:
            detections_by_pair[(int(detection["image_id"]), int(detection.get("category_id", -1)))].append(detection)

    count = len(CGF1_IOU_THRESHOLDS)
    true_positives = [0] * count
    false_positives = [0] * count
    positive_micro_false_positives = [0] * count
    false_negatives = [0] * count
    local_positive_f1s = [0.0] * count
    valid_positive_f1_count = 0
    il_tp = il_fp = il_tn = il_fn = 0

    for image_id in image_ids:
        for category_id in category_ids:
            result = evaluate_cgf1_query(
                gt_by_pair[(image_id, category_id)],
                detections_by_pair[(image_id, category_id)],
                score_threshold,
                iou_type,
                images_by_id[image_id],
            )
            il_tp += int(result["IL_TP"])
            il_fp += int(result["IL_FP"])
            il_tn += int(result["IL_TN"])
            il_fn += int(result["IL_FN"])
            if "TPs" not in result:
                continue
            for index in range(count):
                true_positives[index] += int(result["TPs"][index])
                false_positives[index] += int(result["FPs"][index])
                false_negatives[index] += int(result["FNs"][index])
            if "local_positive_F1s" in result:
                for index in range(count):
                    local_positive_f1s[index] += float(result["local_positive_F1s"][index])
                    positive_micro_false_positives[index] += int(result["FPs"][index])
                if result["num_dt"] > 0:
                    valid_positive_f1_count += 1

    precision = [
        true_positives[index] / (true_positives[index] + false_positives[index] + 1e-4)
        for index in range(count)
    ]
    recall = [
        true_positives[index] / (true_positives[index] + false_negatives[index] + 1e-4)
        for index in range(count)
    ]
    f1 = [
        2.0 * precision[index] * recall[index] / (precision[index] + recall[index] + 1e-4)
        for index in range(count)
    ]
    positive_micro_precision = [
        true_positives[index] / (true_positives[index] + positive_micro_false_positives[index] + 1e-4)
        for index in range(count)
    ]
    positive_micro_f1 = [
        2.0 * positive_micro_precision[index] * recall[index]
        / (positive_micro_precision[index] + recall[index] + 1e-4)
        for index in range(count)
    ]
    positive_macro_f1 = [
        local_positive_f1s[index] / max(valid_positive_f1_count, 1)
        for index in range(count)
    ]
    il_recall = il_tp / (il_tp + il_fn + 1e-6)
    il_precision = il_tp / (il_tp + il_fp + 1e-6)
    il_f1 = 2.0 * il_precision * il_recall / (il_precision + il_recall + 1e-6)
    il_fpr = il_fp / (il_fp + il_tn + 1e-6)
    il_mcc = float(il_tp * il_tn - il_fp * il_fn) / (
        (float(il_tp + il_fp) * float(il_tp + il_fn) * float(il_tn + il_fp) * float(il_tn + il_fn)) ** 0.5
        + 1e-6
    )
    cgf1 = [value * il_mcc for value in positive_micro_f1]
    index_50 = CGF1_IOU_THRESHOLDS.index(0.5)
    index_75 = CGF1_IOU_THRESHOLDS.index(0.75)
    prefix = "bbox" if iou_type == "bbox" else "mask"
    print(f"cgF1 metric, IoU type={iou_type}")
    print(f" Average cgF1              @[ IoU=0.50:0.95] = {mean(cgf1):0.3f}")
    print(f" Average cgF1              @[ IoU=0.50     ] = {cgf1[index_50]:0.3f}")
    print(f" Average cgF1              @[ IoU=0.75     ] = {cgf1[index_75]:0.3f}")
    print(f" Average positive_micro_F1 @[ IoU=0.50:0.95] = {mean(positive_micro_f1):0.3f}")
    print(f" Average IL_MCC                                = {il_mcc:0.3f}")
    return {
        f"cgF1_{prefix}": float(mean(cgf1)),
        f"cgF1_{prefix}50": float(cgf1[index_50]),
        f"cgF1_{prefix}75": float(cgf1[index_75]),
        f"{prefix}_precision": float(mean(precision)),
        f"{prefix}_recall": float(mean(recall)),
        f"{prefix}_F1": float(mean(f1)),
        f"{prefix}_positive_macro_F1": float(mean(positive_macro_f1)),
        f"{prefix}_positive_micro_F1": float(mean(positive_micro_f1)),
        f"{prefix}_IL_precision": float(il_precision),
        f"{prefix}_IL_recall": float(il_recall),
        f"{prefix}_IL_F1": float(il_f1),
        f"{prefix}_IL_FPR": float(il_fpr),
        f"{prefix}_IL_MCC": float(il_mcc),
    }


def load_prediction_items(path):
    payload = load_json(path)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("annotations", "predictions", "results"):
            if isinstance(payload.get(key), list):
                return payload[key]
    raise ValueError("--pred-json must be a COCO result list or contain annotations/predictions/results.")


def print_summary(metrics, num_predictions, args):
    print("\n" + "=" * 80)
    print("LLMDET + SAM MASK EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Predictions evaluated: {num_predictions}")
    print(f"Query mode: {args.query_mode}")
    print("cgF1 protocol: SAM3_LoRA_1 closed-set image-category pairs")
    print("cgF1 score threshold: 0.500")
    print("-" * 80)
    print(f"APmask     : {metrics.get('APmask', 0.0):.4f}")
    print(f"APmask@50  : {metrics.get('APmask50', 0.0):.4f}")
    print(f"APmask@75  : {metrics.get('APmask75', 0.0):.4f}")
    print(f"cgF1_mask  : {metrics.get('cgF1_mask', 0.0):.4f}")
    print(f"cgF1_mask50: {metrics.get('cgF1_mask50', 0.0):.4f}")
    print(f"cgF1_mask75: {metrics.get('cgF1_mask75', 0.0):.4f}")
    if not args.skip_bbox_eval:
        print(f"APbbox     : {metrics.get('APbbox', 0.0):.4f}")
        print(f"cgF1_bbox  : {metrics.get('cgF1_bbox', 0.0):.4f}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Standalone LLMDet + SAM evaluation with COCO APmask and SAM3_LoRA-compatible cgF1."
    )
    parser.add_argument("--val-data-dir", type=str, default=None, help="Directory containing images and _annotations.coco.json.")
    parser.add_argument("--gt-json", type=str, default=None, help="COCO ground-truth annotation JSON.")
    parser.add_argument("--image-root", type=str, default=None, help="Image root; defaults to --val-data-dir or the annotation directory.")
    parser.add_argument("--pred-json", type=str, default=None, help="Existing COCO prediction JSON to evaluate without model inference.")
    parser.add_argument("--output-dir", type=str, default="work_dirs/llmdet_sam_eval", help="Directory for metrics.json and generated predictions.json.")
    parser.add_argument("--output-json", type=str, default=None, help="Metrics JSON path; defaults to <output-dir>/metrics.json.")
    parser.add_argument("--save-pred-json", type=str, default=None, help="Generated prediction JSON path; defaults to <output-dir>/predictions.json.")

    parser.add_argument("--llmdet-repo", type=str, default="LLMDet-main", help="Path to the local LLMDet repository.")
    parser.add_argument("--llmdet-config", type=str, default=None, help="LLMDet MMDetection config; required without --pred-json.")
    parser.add_argument("--llmdet-checkpoint", type=str, default=None, help="LLMDet checkpoint; required without --pred-json.")
    parser.add_argument("--llmdet-chunked-size", type=int, default=-1, help="LLMDet test_cfg.chunked_size override.")
    parser.add_argument("--sam-checkpoint", type=str, default=None, help="SAM checkpoint; required without --pred-json.")
    parser.add_argument("--sam-model-type", choices=["vit_b", "vit_l", "vit_h"], default="vit_b", help="SAM model type.")
    parser.add_argument("--sam-box-batch-size", type=int, default=32, help="Number of detection boxes sent to SAM at once.")
    parser.add_argument("--device", type=str, default="auto", help="auto, cuda, cuda:0, or cpu.")

    parser.add_argument(
        "--query-mode",
        choices=["mmdet-standard", "per-category", "present-categories"],
        default="mmdet-standard",
        help=(
            "mmdet-standard performs one full-class inference per image and preserves "
            "native LLMDet output; per-category and present-categories are legacy "
            "prompt ablations."
        ),
    )
    parser.add_argument(
        "--box-threshold",
        type=float,
        default=None,
        help=(
            "Minimum score sent to SAM. Defaults to 0.0 for mmdet-standard, preserving "
            "COCO AP ranking, and 0.3 for legacy query modes."
        ),
    )
    parser.add_argument("--nms-iou", type=float, default=0.7, help="Legacy per-query box NMS IoU; set negative to disable.")
    parser.add_argument("--max-detections-per-query", type=int, default=100, help="Legacy Top-K boxes per image-category query after NMS.")
    parser.add_argument("--use-sam-score", action="store_true", help="Use detection score multiplied by SAM predicted IoU as final confidence.")

    parser.add_argument("--eval-category-id", type=int, action="append", default=None, help="Evaluate only this category id; can be repeated.")
    parser.add_argument("--eval-category-name", type=str, action="append", default=None, help="Evaluate only this category name; can be repeated.")
    parser.add_argument("--skip-bbox-eval", action="store_true", help="Skip auxiliary bbox AP and bbox cgF1.")
    parser.add_argument("--max-images", type=int, default=None, help="Limit images for debugging; GT is limited to the same subset.")
    parser.add_argument("--log-every", type=int, default=25, help="Progress interval; set 0 to disable.")
    parser.add_argument("--strict", action="store_true", help="Fail on a missing image instead of skipping it.")
    args = parser.parse_args()

    if args.sam_box_batch_size < 1:
        raise ValueError("--sam-box-batch-size must be at least 1.")
    if args.box_threshold is None:
        args.box_threshold = 0.0 if args.query_mode == "mmdet-standard" else 0.3
    if args.nms_iou is not None and args.nms_iou < 0:
        args.nms_iou = None
    if args.query_mode == "present-categories":
        print("WARNING: present-categories is positive-only and does not match SAM3_LoRA_1 all_categories.")
    elif args.query_mode == "per-category":
        print("WARNING: per-category inference is not comparable with MMDet's multi-class evaluation.")
    if not args.pred_json:
        missing = [
            option for option, value in (
                ("--llmdet-config", args.llmdet_config),
                ("--llmdet-checkpoint", args.llmdet_checkpoint),
                ("--sam-checkpoint", args.sam_checkpoint),
            ) if not value
        ]
        if missing:
            raise ValueError(f"Inference requires {', '.join(missing)}.")

    gt_path = resolve_gt_path(args)
    if not gt_path.exists():
        raise FileNotFoundError(f"Ground-truth JSON not found: {gt_path}")
    gt_data = load_json(gt_path)
    eval_category_ids = resolve_eval_category_ids(args, gt_data)
    if eval_category_ids:
        print(f"Filtering evaluation to category ids: {eval_category_ids}")
        gt_data = filter_gt_to_categories(gt_data, eval_category_ids)
    if args.max_images is not None:
        gt_data = limit_gt_to_images(gt_data, args.max_images)
        print(f"Limiting evaluation to the first {len(gt_data['images'])} images.")
    eval_gt = prepare_gt_for_eval(gt_data)
    if not has_gt_segmentation(eval_gt):
        raise ValueError("APmask and mask cgF1 require COCO GT annotations with segmentation.")

    output_dir = Path(args.output_dir)
    output_json = Path(args.output_json) if args.output_json else output_dir / "metrics.json"
    save_pred_json = Path(args.save_pred_json) if args.save_pred_json else output_dir / "predictions.json"
    generated_predictions = not bool(args.pred_json)
    if args.pred_json:
        predictions = load_prediction_items(args.pred_json)
    else:
        predictions = generate_llmdet_sam_predictions(eval_gt, gt_path, args)
        save_pred_json.parent.mkdir(parents=True, exist_ok=True)
        dump_json(save_pred_json, predictions)
        print(f"Saved LLMDet+SAM predictions to {save_pred_json}")
    if not predictions:
        raise RuntimeError("No predictions were generated/evaluated. Try lowering --box-threshold.")

    with tempfile.TemporaryDirectory(prefix="llmdet_sam_eval_") as temporary_directory:
        temporary_directory = Path(temporary_directory)
        gt_file = temporary_directory / "gt.json"
        pred_file = temporary_directory / "pred.json"
        dump_json(gt_file, eval_gt)
        dump_json(pred_file, predictions)
        metrics = {}
        if not args.skip_bbox_eval:
            print("\n" + "=" * 80)
            print("COCO BBOX mAP")
            print("=" * 80)
            metrics.update(run_coco_eval(gt_file, pred_file, "bbox"))
        print("\n" + "=" * 80)
        print("COCO MASK mAP")
        print("=" * 80)
        metrics.update(run_coco_eval(gt_file, pred_file, "segm"))
        if not args.skip_bbox_eval:
            print("\n" + "=" * 80)
            print("BBOX cgF1")
            print("=" * 80)
            metrics.update(run_sam3_lora_compatible_cgf1(eval_gt, predictions, "bbox"))
        print("\n" + "=" * 80)
        print("MASK cgF1")
        print("=" * 80)
        metrics.update(run_sam3_lora_compatible_cgf1(eval_gt, predictions, "segm"))

    query_protocol = {
        "mmdet-standard": "one_full_dataset_category_list_per_image",
        "per-category": "one_query_per_dataset_category_per_image",
        "present-categories": "one_query_per_gt_present_category_per_image",
    }[args.query_mode]
    payload = {
        "gt_json": str(gt_path),
        "pred_json": str(args.pred_json) if args.pred_json else None,
        "save_pred_json": str(save_pred_json) if generated_predictions else None,
        "generated_predictions": generated_predictions,
        "llmdet_repo": str(args.llmdet_repo) if generated_predictions else None,
        "llmdet_config": str(args.llmdet_config) if generated_predictions else None,
        "llmdet_checkpoint": str(args.llmdet_checkpoint) if generated_predictions else None,
        "sam_checkpoint": str(args.sam_checkpoint) if generated_predictions else None,
        "sam_model_type": args.sam_model_type,
        "sam_box_batch_size": args.sam_box_batch_size,
        "query_mode": args.query_mode,
        "query_protocol": query_protocol,
        "box_threshold": args.box_threshold,
        "nms_iou": args.nms_iou if args.query_mode != "mmdet-standard" else None,
        "max_detections_per_query": (
            args.max_detections_per_query if args.query_mode != "mmdet-standard" else None
        ),
        "max_images": args.max_images,
        "use_sam_score": bool(args.use_sam_score),
        "cgf1_protocol": "sam3_lora_1_closed_set_all_categories_compatible",
        "cgf1_score_threshold": 0.5,
        "eval_category_ids": eval_category_ids,
        "num_predictions": len(predictions),
        "metrics": metrics,
    }
    print_summary(metrics, len(predictions), args)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    dump_json(output_json, payload)
    print(f"Saved metrics JSON to {output_json}")


if __name__ == "__main__":
    main()
