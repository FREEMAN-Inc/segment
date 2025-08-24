#!/usr/bin/env python3
import json
import argparse
from pathlib import Path

# Define your class-to-ID mapping here
label2id = {
    "medium": 0,
    "small": 1,
    "large": 2
}

def load_annotations(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # LabelMe style or list of annotations
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "shapes" in data:
        return data["shapes"]
    return [data]

def normalize_points(points, width, height):
    norm = []
    for x, y in points:
        norm.append(x / width)
        norm.append(y / height)
    return norm

def main():
    ap = argparse.ArgumentParser(description="Convert JSON polygons to YOLOv8 segmentation format")
    ap.add_argument("input", help="Input JSON file")
    ap.add_argument("output", help="Output TXT file")
    ap.add_argument("--width", type=float, required=True, help="Image width")
    ap.add_argument("--height", type=float, required=True, help="Image height")
    args = ap.parse_args()

    items = load_annotations(args.input)

    lines = []
    for obj in items:
        label = obj.get("label")
        points = obj.get("points", [])
        if not points or label not in label2id:
            continue

        class_id = label2id[label]
        coords = normalize_points(points, args.width, args.height)

        line = " ".join([str(class_id)] + [f"{v:.6f}" for v in coords])
        lines.append(line)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Wrote {len(lines)} polygons to {args.output}")

if __name__ == "__main__":
    main()
