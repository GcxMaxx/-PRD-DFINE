import json
import os
import random
import shutil
from collections import defaultdict
from typing import Dict, List, Set, Tuple

random.seed(0)


def load_coco(json_path: str) -> Dict:
    with open(json_path, 'r') as f:
        return json.load(f)


def build_index(coco: Dict) -> Tuple[Dict[int, Dict], Dict[int, List[Dict]], Dict[int, Set[int]]]:
    images = {img['id']: img for img in coco['images']}
    anns_by_img: Dict[int, List[Dict]] = defaultdict(list)
    imgs_by_cat: Dict[int, Set[int]] = defaultdict(set)
    for ann in coco['annotations']:
        img_id = ann['image_id']
        anns_by_img[img_id].append(ann)
        imgs_by_cat[ann['category_id']].add(img_id)
    return images, anns_by_img, imgs_by_cat


def stratified_image_subset(
    images: Dict[int, Dict],
    anns_by_img: Dict[int, List[Dict]],
    imgs_by_cat: Dict[int, Set[int]],
    ratio: float,
    min_per_cat: int,
    max_images: int | None = None,
    seed: int = 0,
) -> Set[int]:
    random.seed(seed)
    # desired images per category (by image coverage, not instance count)
    desired: Dict[int, int] = {}
    for cat_id, img_set in imgs_by_cat.items():
        want = int(len(img_set) * ratio)
        desired[cat_id] = max(min_per_cat, want) if ratio > 0 else min_per_cat
        desired[cat_id] = min(desired[cat_id], len(img_set))  # cannot exceed available

    # Underfilled counter
    have: Dict[int, int] = defaultdict(int)

    # For each image, compute which categories it contributes to
    cats_per_img: Dict[int, Set[int]] = {
        img_id: set(ann['category_id'] for ann in anns)
        for img_id, anns in anns_by_img.items()
    }

    # Candidate pool
    all_img_ids = list(images.keys())
    random.shuffle(all_img_ids)

    selected: Set[int] = set()

    def benefit(img_id: int) -> int:
        # how many underfilled categories would this image help
        b = 0
        for c in cats_per_img.get(img_id, set()):
            if have[c] < desired.get(c, 0):
                b += 1
        return b

    # Greedy: iterate until all categories reached desired or cap reached
    unmet = lambda: any(have[c] < desired[c] for c in desired)

    # Multi-pass: first pass pick most beneficial images
    # To keep complexity reasonable, do a few rounds of best-first selection
    rounds = 3
    for _ in range(rounds):
        all_img_ids.sort(key=lambda x: benefit(x), reverse=True)
        for img_id in list(all_img_ids):
            if max_images is not None and len(selected) >= max_images:
                break
            b = benefit(img_id)
            if b <= 0:
                continue
            selected.add(img_id)
            all_img_ids.remove(img_id)
            # update have
            for c in cats_per_img.get(img_id, set()):
                if have[c] < desired.get(c, 0):
                    have[c] += 1
        if not unmet():
            break

    # Fallback: random fill until we meet or we run out
    for img_id in list(all_img_ids):
        if max_images is not None and len(selected) >= max_images:
            break
        if not unmet():
            break
        b = benefit(img_id)
        if b > 0:
            selected.add(img_id)
            for c in cats_per_img.get(img_id, set()):
                if have[c] < desired.get(c, 0):
                    have[c] += 1

    return selected


essential_image_fields = [
    'id', 'file_name', 'width', 'height', 'license', 'date_captured'
]

def filter_coco(coco: Dict, keep_img_ids: Set[int]) -> Dict:
    img_set = set(keep_img_ids)
    new_images = [img for img in coco['images'] if img['id'] in img_set]
    new_anns = [ann for ann in coco['annotations'] if ann['image_id'] in img_set]

    # Optionally, you can drop categories not present in subset; here we keep all to preserve mapping
    new_coco = {
        'images': new_images,
        'annotations': new_anns,
        'categories': coco['categories'],
        'licenses': coco.get('licenses', []),
        'info': coco.get('info', {}),
    }
    return new_coco


def copy_images(images: List[Dict], src_dir: str, dst_dir: str, symlink: bool = False):
    os.makedirs(dst_dir, exist_ok=True)
    for img in images:
        src = os.path.join(src_dir, img['file_name'])
        dst = os.path.join(dst_dir, img['file_name'])
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if not os.path.exists(src):
            # fallback: try basename only if subfolders present in file_name
            src_alt = os.path.join(src_dir, os.path.basename(img['file_name']))
            if os.path.exists(src_alt):
                src = src_alt
        if symlink:
            try:
                if os.path.lexists(dst):
                    os.remove(dst)
                os.symlink(src, dst)
            except Exception:
                shutil.copy2(src, dst)
        else:
            shutil.copy2(src, dst)


def summarize(coco: Dict) -> Dict[int, int]:
    per_cat = defaultdict(int)
    for ann in coco['annotations']:
        per_cat[ann['category_id']] += 1
    return dict(sorted(per_cat.items()))


def run(
    train_json: str,
    val_json: str,
    images_train_dir: str,
    images_val_dir: str,
    out_train_dir: str,
    out_val_dir: str,
    out_ann_dir: str,
    train_ratio: float = 0.1,
    val_ratio: float = 0.1,
    min_per_cat: int = 20,
    seed: int = 42,
    symlink: bool = False,
):
    os.makedirs(out_ann_dir, exist_ok=True)

    # Train subset
    coco_tr = load_coco(train_json)
    imgs_tr, anns_by_img_tr, imgs_by_cat_tr = build_index(coco_tr)
    keep_tr = stratified_image_subset(imgs_tr, anns_by_img_tr, imgs_by_cat_tr, train_ratio, min_per_cat, seed=seed)
    sub_tr = filter_coco(coco_tr, keep_tr)

    # Val subset
    coco_val = load_coco(val_json)
    imgs_val, anns_by_img_val, imgs_by_cat_val = build_index(coco_val)
    keep_val = stratified_image_subset(imgs_val, anns_by_img_val, imgs_by_cat_val, val_ratio, min_per_cat, seed=seed+1)
    sub_val = filter_coco(coco_val, keep_val)

    # Copy images
    copy_images(sub_tr['images'], images_train_dir, out_train_dir, symlink=symlink)
    copy_images(sub_val['images'], images_val_dir, out_val_dir, symlink=symlink)

    # Write JSONs
    out_tr_json = os.path.join(out_ann_dir, 'instances_train_cat0_16_subset.json')
    out_val_json = os.path.join(out_ann_dir, 'instances_val_cat0_16_subset.json')
    with open(out_tr_json, 'w') as f:
        json.dump(sub_tr, f)
    with open(out_val_json, 'w') as f:
        json.dump(sub_val, f)

    # Summaries
    print('[Train] images:', len(sub_tr['images']), 'anns:', len(sub_tr['annotations']))
    print('[Val]   images:', len(sub_val['images']), 'anns:', len(sub_val['annotations']))
    print('[Train] per-cat instances:', summarize(sub_tr))
    print('[Val]   per-cat instances:', summarize(sub_val))
    print('Wrote:', out_tr_json, out_val_json)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-json', default='dataset/InsPLAD-det/annotations/instances_train_cat0_16.json')
    parser.add_argument('--val-json', default='dataset/InsPLAD-det/annotations/instances_val_cat0_16.json')
    parser.add_argument('--images-train', default='dataset/InsPLAD-det/train')
    parser.add_argument('--images-val', default='dataset/InsPLAD-det/val')
    parser.add_argument('--out-train', default='dataset/InsPLAD-det/train1')
    parser.add_argument('--out-val', default='dataset/InsPLAD-det/val1')
    parser.add_argument('--out-ann', default='dataset/InsPLAD-det/annotations1')
    parser.add_argument('--train-ratio', type=float, default=0.1)
    parser.add_argument('--val-ratio', type=float, default=0.1)
    parser.add_argument('--min-per-cat', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--symlink', action='store_true')
    args = parser.parse_args()

    run(
        train_json=args.train_json,
        val_json=args.val_json,
        images_train_dir=args.images_train,
        images_val_dir=args.images_val,
        out_train_dir=args.out_train,
        out_val_dir=args.out_val,
        out_ann_dir=args.out_ann,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        min_per_cat=args.min_per_cat,
        seed=args.seed,
        symlink=args.symlink,
    )
