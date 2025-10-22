from pathlib import Path
from tqdm import tqdm
from PIL import Image
import shutil, os, random, re

def slugify(x: str) -> str:
    return re.sub(r'[^a-z0-9_\-]', '_', x.lower())

def copy_and_resize(src_img, dst_img, size=(224,224)):
    img = Image.open(src_img).convert("RGB")
    img = img.resize(size, Image.BICUBIC)
    dst_img.parent.mkdir(parents=True, exist_ok=True)
    img.save(dst_img, quality=95)

def stratified_split(image_paths, labels, train=0.8, val=0.1, test=0.1, seed=42):
    random.seed(seed)
    by_label = {}
    for p, y in zip(image_paths, labels):
        by_label.setdefault(y, []).append(p)
    splits = {"train":[], "val":[], "test":[]}
    for y, items in by_label.items():
        random.shuffle(items)
        n = len(items)
        n_train = int(n*train); n_val = int(n*val)
        splits["train"] += [(p,y) for p in items[:n_train]]
        splits["val"]   += [(p,y) for p in items[n_train:n_train+n_val]]
        splits["test"]  += [(p,y) for p in items[n_train+n_val:]]
    return splits

def unify_raw_to_imagefolder(raw_root: Path, out_root: Path, image_size=(224,224)):
    # Expect structure: raw_root/<dataset>/<class>/*.jpg
    image_paths, labels = [], []
    for dataset_dir in raw_root.iterdir():
        if not dataset_dir.is_dir(): continue
        for cls_dir in dataset_dir.glob("*"):
            if not cls_dir.is_dir(): continue
            label = slugify(cls_dir.name)
            for img in cls_dir.rglob("*.jpg"):
                image_paths.append(img)
                labels.append(label)
            for img in cls_dir.rglob("*.png"):
                image_paths.append(img)
                labels.append(label)

    splits = stratified_split(image_paths, labels)
    for split, items in splits.items():
        for p, y in tqdm(items, desc=f"Writing {split}"):
            dst = out_root / split / y / p.name
            copy_and_resize(p, dst, size=image_size)

if __name__ == "__main__":
    raw = Path("data/raw")
    out = Path("data/processed")
    unify_raw_to_imagefolder(raw, out, image_size=(224,224))
    print("Wrote ImageFolder to", out)
