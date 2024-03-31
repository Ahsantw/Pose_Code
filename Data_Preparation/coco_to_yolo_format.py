import pandas as pd
from utils import *
import json
from collections import defaultdict
import os
def convert_coco_json(json_dir="../coco/annotations/", use_segments=False, cls91to80=False):
    """Converts COCO JSON format to YOLO label format, with options for segments and class mapping."""
    save_dir = make_dirs()  # output directory
    coco80 = coco91_to_coco80_class()

    # Import json
    for json_file in sorted(Path(json_dir).resolve().glob("*.json")):
        fn = Path(save_dir) / "labels" / json_file.stem.replace("instances_", "")  # folder name
        fn.mkdir()
        with open(json_file) as f:
            data = json.load(f)

        # Create image dict
        images = {"%g" % x["id"]: x for x in data["images"]}
        # Create image-annotations dict
        imgToAnns = defaultdict(list)
        for ann in data["annotations"]:
            imgToAnns[ann["image_id"]].append(ann)

        # Write labels file
        for img_id, anns in tqdm(imgToAnns.items(), desc=f"Annotations {json_file}"):
            img = images["%g" % img_id]
            h, w, f = img["height"], img["width"], img["file_name"]

            bboxes = []
            segments = []
            for ann in anns:
                if ann["iscrowd"]:
                    continue
                # The COCO box format is [top left x, top left y, width, height]
                box = np.array(ann["bbox"], dtype=np.float64)
                box[:2] += box[2:] / 2  # xy top-left corner to center
                box[[0, 2]] /= w  # normalize x
                box[[1, 3]] /= h  # normalize y
                if box[2] <= 0 or box[3] <= 0:  # if w <= 0 and h <= 0
                    continue

                cls = coco80[ann["category_id"] - 1] if cls91to80 else ann["category_id"] - 1  # class
                box = [cls] + box.tolist()
                if box not in bboxes:
                    bboxes.append(box)
                # Segments
                if use_segments:
                    if len(ann["segmentation"]) > 1:
                        s = merge_multi_segment(ann["segmentation"])
                        s = (np.concatenate(s, axis=0) / np.array([w, h])).reshape(-1).tolist()
                    else:
                        s = [j for i in ann["segmentation"] for j in i]  # all segments concatenated
                        s = (np.array(s).reshape(-1, 2) / np.array([w, h])).reshape(-1).tolist()
                    s = [cls] + s
                    if s not in segments:
                        segments.append(s)

            # Write
            with open((fn / f).with_suffix(".txt"), "a") as file:
                for i in range(len(bboxes)):
                    line = (*(segments[i] if use_segments else bboxes[i]),)  # cls, box or segments
                    file.write(("%g " * len(line)).rstrip() % line + "\n")

os.system('rm -rf annotations/')
os.makedirs('annotations',exist_ok=True)
os.system('cp *.json annotations/')
convert_coco_json(
            "annotations",  # directory with *.json
            use_segments=False,
            cls91to80=False)