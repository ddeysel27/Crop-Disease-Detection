import json, random, os
from pathlib import Path

def set_seed(seed: int = 42):
    import numpy as np, torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def save_label_mapping(class_to_idx, out_path):
    idx2label = {int(v): k for k, v in class_to_idx.items()}
    Path(out_path).write_text(json.dumps(idx2label, indent=2))
