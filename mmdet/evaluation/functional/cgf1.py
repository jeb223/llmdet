# Copyright (c) OpenMMLab. All rights reserved.
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple


CGF1_IOU_THRESHOLDS = tuple(round(0.5 + 0.05 * i, 2) for i in range(10))


def _bbox_area_xywh(box: Sequence[float]) -> float:
    return max(0.0, float(box[2])) * max(0.0, float(box[3]))


def _bbox_iou_xywh(dt_box: Sequence[float],
                   gt_box: Sequence[float],
                   gt_iscrowd: bool = False) -> float:
    dx1, dy1, dw, dh = [float(x) for x in dt_box]
    gx1, gy1, gw, gh = [float(x) for x in gt_box]
    dx2, dy2 = dx1 + dw, dy1 + dh
    gx2, gy2 = gx1 + gw, gy1 + gh
    ix1, iy1 = max(dx1, gx1), max(dy1, gy1)
    ix2, iy2 = min(dx2, gx2), min(dy2, gy2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if intersection <= 0:
        return 0.0

    dt_area = _bbox_area_xywh(dt_box)
    gt_area = _bbox_area_xywh(gt_box)
    denominator = dt_area if gt_iscrowd else dt_area + gt_area - intersection
    if denominator <= 0:
        return 0.0
    return min(1.0, intersection / denominator)


def _hungarian_minimize(costs: Sequence[Sequence[float]]) -> List[Tuple[int,
                                                                         int]]:
    num_rows = len(costs)
    num_cols = len(costs[0]) if num_rows else 0
    if num_rows == 0 or num_cols == 0:
        return []
    if num_rows > num_cols:
        raise ValueError('_hungarian_minimize requires rows <= columns')

    u = [0.0] * (num_rows + 1)
    v = [0.0] * (num_cols + 1)
    p = [0] * (num_cols + 1)
    way = [0] * (num_cols + 1)

    for row in range(1, num_rows + 1):
        p[0] = row
        col0 = 0
        min_values = [float('inf')] * (num_cols + 1)
        used = [False] * (num_cols + 1)
        while True:
            used[col0] = True
            row0 = p[col0]
            delta = float('inf')
            col1 = 0
            for col in range(1, num_cols + 1):
                if used[col]:
                    continue
                current = costs[row0 - 1][col - 1] - u[row0] - v[col]
                if current < min_values[col]:
                    min_values[col] = current
                    way[col] = col0
                if min_values[col] < delta:
                    delta = min_values[col]
                    col1 = col
            for col in range(num_cols + 1):
                if used[col]:
                    u[p[col]] += delta
                    v[col] -= delta
                else:
                    min_values[col] -= delta
            col0 = col1
            if p[col0] == 0:
                break
        while True:
            col1 = way[col0]
            p[col0] = p[col1]
            col0 = col1
            if col0 == 0:
                break

    assignments = []
    for col in range(1, num_cols + 1):
        if p[col] != 0:
            assignments.append((p[col] - 1, col - 1))
    return assignments


def _max_weight_assignment(
        weights: Sequence[Sequence[float]]) -> List[Tuple[int, int]]:
    num_rows = len(weights)
    num_cols = len(weights[0]) if num_rows else 0
    if num_rows == 0 or num_cols == 0:
        return []
    if num_rows <= num_cols:
        costs = [[1.0 - float(value) for value in row] for row in weights]
        return _hungarian_minimize(costs)
    costs = [[
        1.0 - float(weights[row][col]) for row in range(num_rows)
    ] for col in range(num_cols)]
    matches = _hungarian_minimize(costs)
    return [(dt_idx, gt_idx) for gt_idx, dt_idx in matches]


def _f1_from_counts(tp: int, fp: int, fn: int) -> float:
    precision = tp / (tp + fp + 1e-4)
    recall = tp / (tp + fn + 1e-4)
    return 2.0 * precision * recall / (precision + recall + 1e-4)


def _evaluate_cgf1_query(gt_anns: Sequence[dict], dt_anns: Sequence[dict],
                         score_threshold: float) -> dict:
    gt = [
        ann for ann in gt_anns
        if not bool(ann.get('ignore', 0))
        and not bool(ann.get('iscrowd', 0))
    ]
    dt = [
        ann for ann in dt_anns
        if float(ann.get('score', 0.0)) >= score_threshold
    ]
    result = dict(IL_TP=0, IL_TN=0, IL_FP=0, IL_FN=0, num_dt=len(dt))

    if len(gt) == 0 and len(dt) == 0:
        result['IL_TN'] = 1
        return result

    if len(gt) > 0 and len(dt) == 0:
        result['IL_FN'] = 1
        result['TPs'] = [0] * len(CGF1_IOU_THRESHOLDS)
        result['FPs'] = [0] * len(CGF1_IOU_THRESHOLDS)
        result['FNs'] = [len(gt)] * len(CGF1_IOU_THRESHOLDS)
        result['local_positive_F1s'] = [0.0] * len(CGF1_IOU_THRESHOLDS)
        return result

    ious = [[
        _bbox_iou_xywh(det['bbox'], ann['bbox'], bool(ann.get('iscrowd', 0)))
        for ann in gt
    ] for det in dt]
    matches = _max_weight_assignment(ious)
    match_scores = [ious[dt_idx][gt_idx] for dt_idx, gt_idx in matches]

    tps, fps, fns, local_f1s = [], [], [], []
    for iou_threshold in CGF1_IOU_THRESHOLDS:
        tp = sum(1 for score in match_scores if score >= iou_threshold)
        fp = len(dt) - tp
        fn = len(gt) - tp
        tps.append(tp)
        fps.append(fp)
        fns.append(fn)
        local_f1s.append(_f1_from_counts(tp, fp, fn))

    result.update(
        IL_TP=int(len(gt) > 0 and len(dt) > 0),
        IL_FP=int(len(gt) == 0 and len(dt) > 0),
        IL_TN=int(len(gt) == 0 and len(dt) == 0),
        IL_FN=int(len(gt) > 0 and len(dt) == 0),
        TPs=tps,
        FPs=fps,
        FNs=fns)
    if len(gt) > 0 and len(dt) > 0:
        result['local_positive_F1s'] = local_f1s
    return result


def _mean(values: Sequence[float]) -> float:
    return sum(values) / max(len(values), 1)


def eval_bbox_cgf1(gt_data: dict,
                   pred_data: Sequence[dict],
                   image_ids: Sequence[int],
                   cat_ids: Sequence[int],
                   score_threshold: float = 0.5,
                   use_cats: bool = True) -> Dict[str, float]:
    """Evaluate category-grounded F1 on COCO-format bounding boxes.

    This follows the cgF1 calculation used by the project's standalone
    SAM3-style LLMDet evaluator. It computes per-query positive micro F1 over
    IoU thresholds and scales it by the image-level Matthews correlation
    coefficient for positive/negative category-query decisions.
    """
    image_infos = {
        int(image['id']): image
        for image in gt_data.get('images', [])
    }
    eval_image_ids = [
        int(image_id) for image_id in image_ids
        if bool(image_infos.get(int(image_id), {}).get(
            'is_instance_exhaustive', True))
    ]
    eval_cat_ids = [int(cat_id) for cat_id in cat_ids] if use_cats else [-1]
    if use_cats and not eval_cat_ids:
        eval_cat_ids = [1]

    gt_by_query = defaultdict(list)
    for ann in gt_data.get('annotations', []):
        image_id = int(ann['image_id'])
        if image_id not in eval_image_ids:
            continue
        category_id = int(ann.get('category_id', -1)) if use_cats else -1
        gt_by_query[(image_id, category_id)].append(ann)

    dt_by_query = defaultdict(list)
    for pred in pred_data:
        image_id = int(pred['image_id'])
        if image_id not in eval_image_ids:
            continue
        category_id = int(pred.get('category_id', -1)) if use_cats else -1
        dt_by_query[(image_id, category_id)].append(pred)

    tps = [0] * len(CGF1_IOU_THRESHOLDS)
    fps = [0] * len(CGF1_IOU_THRESHOLDS)
    positive_micro_fps = [0] * len(CGF1_IOU_THRESHOLDS)
    fns = [0] * len(CGF1_IOU_THRESHOLDS)
    local_positive_f1s = [0.0] * len(CGF1_IOU_THRESHOLDS)
    valid_positive_f1_count = 0
    il_tp = il_fp = il_tn = il_fn = 0

    for image_id in eval_image_ids:
        for category_id in eval_cat_ids:
            result = _evaluate_cgf1_query(
                gt_by_query[(image_id, category_id)],
                dt_by_query[(image_id, category_id)],
                score_threshold=score_threshold)
            il_tp += int(result['IL_TP'])
            il_fp += int(result['IL_FP'])
            il_tn += int(result['IL_TN'])
            il_fn += int(result['IL_FN'])

            if 'TPs' not in result:
                continue
            for index in range(len(CGF1_IOU_THRESHOLDS)):
                tps[index] += int(result['TPs'][index])
                fps[index] += int(result['FPs'][index])
                fns[index] += int(result['FNs'][index])
            if 'local_positive_F1s' in result:
                for index in range(len(CGF1_IOU_THRESHOLDS)):
                    local_positive_f1s[index] += float(
                        result['local_positive_F1s'][index])
                    positive_micro_fps[index] += int(result['FPs'][index])
                if result['num_dt'] > 0:
                    valid_positive_f1_count += 1

    precision = [
        tps[i] / (tps[i] + fps[i] + 1e-4)
        for i in range(len(CGF1_IOU_THRESHOLDS))
    ]
    recall = [
        tps[i] / (tps[i] + fns[i] + 1e-4)
        for i in range(len(CGF1_IOU_THRESHOLDS))
    ]
    f1 = [
        2.0 * precision[i] * recall[i] /
        (precision[i] + recall[i] + 1e-4)
        for i in range(len(CGF1_IOU_THRESHOLDS))
    ]
    positive_micro_precision = [
        tps[i] / (tps[i] + positive_micro_fps[i] + 1e-4)
        for i in range(len(CGF1_IOU_THRESHOLDS))
    ]
    positive_micro_f1 = [
        2.0 * positive_micro_precision[i] * recall[i] /
        (positive_micro_precision[i] + recall[i] + 1e-4)
        for i in range(len(CGF1_IOU_THRESHOLDS))
    ]
    positive_macro_f1 = [
        local_positive_f1s[i] / max(valid_positive_f1_count, 1)
        for i in range(len(CGF1_IOU_THRESHOLDS))
    ]

    il_recall = il_tp / (il_tp + il_fn + 1e-6)
    il_precision = il_tp / (il_tp + il_fp + 1e-6)
    il_f1 = 2.0 * il_precision * il_recall / (
        il_precision + il_recall + 1e-6)
    il_fpr = il_fp / (il_fp + il_tn + 1e-6)
    il_mcc_denominator = (
        float(il_tp + il_fp) * float(il_tp + il_fn) *
        float(il_tn + il_fp) * float(il_tn + il_fn))**0.5 + 1e-6
    il_mcc = float(il_tp * il_tn - il_fp * il_fn) / il_mcc_denominator
    cgf1 = [value * il_mcc for value in positive_micro_f1]

    index50 = CGF1_IOU_THRESHOLDS.index(0.5)
    index75 = CGF1_IOU_THRESHOLDS.index(0.75)
    return {
        'cgF1_bbox': float(_mean(cgf1)),
        'cgF1_bbox50': float(cgf1[index50]),
        'cgF1_bbox75': float(cgf1[index75]),
        'bbox_precision': float(_mean(precision)),
        'bbox_recall': float(_mean(recall)),
        'bbox_F1': float(_mean(f1)),
        'bbox_positive_macro_F1': float(_mean(positive_macro_f1)),
        'bbox_positive_micro_F1': float(_mean(positive_micro_f1)),
        'bbox_IL_precision': float(il_precision),
        'bbox_IL_recall': float(il_recall),
        'bbox_IL_F1': float(il_f1),
        'bbox_IL_FPR': float(il_fpr),
        'bbox_IL_MCC': float(il_mcc),
    }
