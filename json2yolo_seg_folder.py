#!/usr/bin/env python3
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Optional: use Pillow to read image size if JSON doesn't include it.
try:
    from PIL import Image
except Exception:
    Image = None

def parse_class_map(s: Optional[str], fp: Optional[str]) -> Dict[str, int]:
    """
    Build label->id map from either:
      --class-map "medium:0,small:1,large:2"
    or --class-map-json path/to/map.json  (e.g., {"medium":0,"small":1})
    """
    if fp:
        with open(fp, "r", encoding="utf-8") as f:
            m = json.load(f)
        return {str(k): int(v) for k, v in m.items()}
    if s:
        m = {}
        for pair in s.split(","):
            if not pair.strip():
                continue
            k, v = pair.split(":")
            m[k.strip()] = int(v.strip())
        return m
    # default single class
    return {"medium": 0}

def load_shapes(obj) -> List[dict]:
    """
    Accept common forms:
      1) LabelMe: {"shapes":[{label, points, shape_type, ...}], "imageWidth":W, "imageHeight":H}
      2) A single annotation dict (has label/points)
      3) A list of such dicts
    Return a list of {label, points, shape_type} dicts.
    """
    if isinstance(obj, dict):
        if "shapes" in obj and isinstance(obj["shapes"], list):
            return obj["shapes"]
        # maybe a single object shaped like a shape
        if "points" in obj:
            return [obj]
        return []
    elif isinstance(obj, list):
        # list of shapes
        return obj
    return []

def get_image_size_from_json(obj) -> Tuple[Optional[int], Optional[int]]:
    if isinstance(obj, dict):
        W = obj.get("imageWidth")
        H = obj.get("imageHeight")
        if isinstance(W, int) and isinstance(H, int):
            return W, H
    return None, None

def find_image_size_from_disk(json_path: Path) -> Tuple[Optional[int], Optional[int]]:
    if Image is None:
        return None, None
    # Try common image extensions with same stem
    for ext in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"):
        img_path = json_path.with_suffix(ext)
        if img_path.exists():
            try:
                with Image.open(img_path) as im:
                    w, h = im.size
                return w, h
            except Exception:
                pass
    return None, None

def normalize_points(points: List[List[float]], W: float, H: float) -> List[float]:
    out = []
    for xy in points:
        if len(xy) != 2:
            continue
        x, y = xy
        nx = x / W
        ny = y / H
        # clamp to [0,1] just in case
        nx = 0.0 if nx < 0 else (1.0 if nx > 1 else nx)
        ny = 0.0 if ny < 0 else (1.0 if ny > 1 else ny)
        out.extend([nx, ny])
    return out

def write_yolo_seg(lines: List[str], out_file: Path):
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

def process_json(
    json_file: Path,
    out_dir: Path,
    class_map: Dict[str, int],
    decimals: int = 6
):
    with open(json_file, "r", encoding="utf-8") as f:
        obj = json.load(f)

    shapes = load_shapes(obj)
    if not shapes:
        return

    # Determine image size (required for YOLO normalized coords)
    W, H = get_image_size_from_json(obj)
    if not (W and H):
        W, H = find_image_size_from_disk(json_file)
    if not (W and H):
        # Heuristic: try to read size from a sibling text/json meta (optional)
        # Or fail loudly:
        print(f"[WARN] No image size for {json_file.name}; skipping.")
        return

    lines = []
    for sh in shapes:
        label = sh.get("label") or sh.get("category") or ""
        if label not in class_map:
            # skip unknown labels to avoid wrong IDs
            # (or you can auto-assign new IDs here if desired)
            continue

        if sh.get("shape_type", "polygon") != "polygon":
            continue  # YOLOv8-seg expects polygons

        points = sh.get("points", [])
        if not points:
            continue

        coords = normalize_points(points, W, H)
        if not coords:
            continue

        class_id = class_map[label]
        line = " ".join([str(class_id)] + [f"{v:.{decimals}f}" for v in coords])
        lines.append(line)

    if not lines:
        return

    # YOLO expects a .txt per image; use same stem as the JSON
    out_path = out_dir / f"{json_file.stem}.txt"
    write_yolo_seg(lines, out_path)
    print(f"[OK] {json_file} -> {out_path} ({len(lines)} objs)")

def main():
    ap = argparse.ArgumentParser(
        description="Iterate a folder of JSONs and emit YOLOv8-seg label .txt files (one per image)."
    )
    ap.add_argument("--input-dir", required=True, help="Directory containing JSON annotations")
    ap.add_argument("--output-dir", required=True, help="Directory to write YOLO label .txt files")
    ap.add_argument("--class-map", default=None,
                    help='Inline map like "medium:0,small:1,large:2"')
    ap.add_argument("--class-map-json", default=None,
                    help="Path to a JSON file with label->id mapping")
    ap.add_argument("--decimals", type=int, default=6, help="Rounding for normalized coords")
    args = ap.parse_args()

    class_map = parse_class_map(args.class_map, args.class_map_json)
    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)

    for jf in in_dir.rglob("*.json"):
        process_json(jf, out_dir, class_map, decimals=args.decimals)

if __name__ == "__main__":
    main()
