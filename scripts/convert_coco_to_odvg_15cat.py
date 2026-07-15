#!/usr/bin/env python3
"""Convert flat COCO splits to LLMDet ODVG JSONL.

Expected input layout:
  data/train/_annotations.coco.json
  data/valid/_annotations.coco.json
  data/train/*.jpg
  data/valid/*.jpg

Outputs:
  data/odvg_label_map.json
  data/train/odvg_train.jsonl
  data/valid/odvg_valid.jsonl
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def xywh_to_xyxy(box, width, height):
    x, y, w, h = [float(v) for v in box]
    x1 = max(0.0, min(x, float(width)))
    y1 = max(0.0, min(y, float(height)))
    x2 = max(0.0, min(x + w, float(width)))
    y2 = max(0.0, min(y + h, float(height)))
    return [x1, y1, x2, y2]


def normalize_name(name):
    return " ".join(str(name).strip().lower().split())


def build_label_maps(coco):
    cats = sorted(coco["categories"], key=lambda c: int(c["id"]))
    cat_id_to_label = {int(cat["id"]): idx for idx, cat in enumerate(cats)}
    label_map = {str(idx): normalize_name(cat["name"]) for idx, cat in enumerate(cats)}
    return cat_id_to_label, label_map


def convert_split(data_root, split, label_map, cat_id_to_label):
    ann_path = data_root / split / "_annotations.coco.json"
    out_path = data_root / split / f"odvg_{split}.jsonl"

    coco = json.loads(ann_path.read_text(encoding="utf-8"))
    images = {int(img["id"]): img for img in coco["images"]}
    anns_by_image = defaultdict(list)
    skipped = 0

    for ann in coco["annotations"]:
        image_id = int(ann["image_id"])
        category_id = int(ann["category_id"])
        if image_id not in images or category_id not in cat_id_to_label:
            skipped += 1
            continue
        img = images[image_id]
        box = xywh_to_xyxy(ann["bbox"], img["width"], img["height"])
        if box[2] - box[0] < 1 or box[3] - box[1] < 1:
            skipped += 1
            continue
        anns_by_image[image_id].append({
            "bbox": box,
            "label": cat_id_to_label[category_id],
        })

    written_images = 0
    written_boxes = 0
    with out_path.open("w", encoding="utf-8") as f:
        for image_id in sorted(images):
            img = images[image_id]
            instances = anns_by_image.get(image_id, [])
            present_labels = sorted({obj["label"] for obj in instances})
            present_names = [label_map[str(label)] for label in present_labels]
            if present_names:
                answer = "This image contains " + ", ".join(present_names) + "."
            else:
                answer = "This image contains no target objects."

            record = {
                "filename": img["file_name"],
                "height": int(img["height"]),
                "width": int(img["width"]),
                "detection": {"instances": instances},
                "tags": present_names,
                "conversations": [
                    {
                        "from": "human",
                        "value": "<image>\nDescribe the image in detail.",
                    },
                    {"from": "gpt", "value": answer},
                ],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written_images += 1
            written_boxes += len(instances)

    return {
        "split": split,
        "images": written_images,
        "boxes": written_boxes,
        "skipped_annotations": skipped,
        "output": str(out_path),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data", help="LLMDet data root.")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    train_coco = json.loads((data_root / "train" / "_annotations.coco.json").read_text(encoding="utf-8"))
    cat_id_to_label, label_map = build_label_maps(train_coco)

    label_map_path = data_root / "odvg_label_map.json"
    label_map_path.write_text(json.dumps(label_map, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    summaries = [
        convert_split(data_root, "train", label_map, cat_id_to_label),
        convert_split(data_root, "valid", label_map, cat_id_to_label),
    ]

    print(f"Wrote label map: {label_map_path}")
    print("Classes:")
    for key, value in label_map.items():
        print(f"  {key}: {value}")
    for item in summaries:
        print(
            f"{item['split']}: images={item['images']} boxes={item['boxes']} "
            f"skipped={item['skipped_annotations']} output={item['output']}"
        )


if __name__ == "__main__":
    main()
