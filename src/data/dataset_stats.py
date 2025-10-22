from pathlib import Path
from collections import Counter

def count_classes(root="data/processed/train"):
    root = Path(root)
    counts = {p.name: len(list(p.glob("*.jpg"))) for p in root.iterdir() if p.is_dir()}
    return dict(sorted(counts.items(), key=lambda kv: kv[0]))

if __name__ == "__main__":
    print(count_classes())
