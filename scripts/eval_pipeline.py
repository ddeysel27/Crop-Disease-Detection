#!/usr/bin/env python

"""
Example usage:

    python scripts/eval_pipeline.py \
        --dataset data/processed/cassava/val \
        --species cassava
"""

import argparse
from pathlib import Path
from typing import List

import json
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, classification_report

import torch

from app.config import (
    MODELS_DIR,
    DEVICE,
)
from src.load_model.loaders import (
    load_yolo_seg_model,
    load_vit_model,
)
from src.pipeline.inference import run_pipeline


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, required=True,
                  help="Path to dataset root with class subfolders (val/test).")
    p.add_argument("--species", type=str, required=True,
                  help="Assumed species for this dataset (e.g. cassava, rice).")
    p.add_argument("--output", type=str, default=None,
                  help="Optional path to save JSON with metrics.")
    return p.parse_args()


def main():
    args = parse_args()

    dataset_root = Path(args.dataset)
    assumed_species = args.species

    # ----- label mappings (keep in sync with notebook) -----
    SPECIES_LABELS = ["cassava", "rice", "plantvillage"]
    SPECIES_IDX2NAME = {i: name for i, name in enumerate(SPECIES_LABELS)}

    CASSAVA_LABELS = [
        "cassava_bacterial_blight",
        "cassava_brown_streak",
        "cassava_green_mottle",
        "cassava_healthy",
    ]
    RICE_LABELS = [
        "rice_bacterial_leaf_blight",
        "rice_brown_spot",
        "rice_leaf_smut",
    ]
    PLANTV_LABELS = [
        "tomato_bacterial_spot",
        "tomato_early_blight",
        "tomato_healthy",
        # ...
    ]
    DISEASE_LABEL_MAP = {
        "cassava": CASSAVA_LABELS,
        "rice": RICE_LABELS,
        "plantvillage": PLANTV_LABELS,
    }

    # ----- load models -----
    yolo = load_yolo_seg_model(MODELS_DIR / "yolo_plantdoc_seg.pt").to(DEVICE)
    species_model = load_vit_model(MODELS_DIR / "species_classifier_vit.pth").to(DEVICE)
    cassava_model = load_vit_model(MODELS_DIR / "cassava_best.pth").to(DEVICE)
    rice_model    = load_vit_model(MODELS_DIR / "rice_best.pth").to(DEVICE)
    plantv_model  = load_vit_model(MODELS_DIR / "plant_village_best.pth").to(DEVICE)

    disease_models = {
        "cassava": cassava_model,
        "rice": rice_model,
        "plantvillage": plantv_model,
    }

    # ----- walk dataset -----
    img_paths: List[Path] = []
    y_true: List[str] = []

    for class_dir in sorted(dataset_root.iterdir()):
        if not class_dir.is_dir():
            continue
        label = class_dir.name
        for p in class_dir.rglob("*"):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                img_paths.append(p)
                y_true.append(label)

    y_pred: List[str] = []

    for img_path in tqdm(img_paths, desc=f"Evaluating {dataset_root}"):
        species_pred, disease_pred = run_pipeline(
            img_path,
            yolo_model=yolo,
            species_model=species_model,
            species_idx2name=SPECIES_IDX2NAME,
            disease_models=disease_models,
            disease_label_map=DISEASE_LABEL_MAP,
        )

        # You can check species_pred vs assumed_species here if you want
        y_pred.append(disease_pred)

    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, zero_division=0, output_dict=True)

    print(f"\nAccuracy: {acc:.4f}\n")
    print("Per-class report:")
    from pprint import pprint
    pprint(report)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(
                {
                    "dataset": str(dataset_root),
                    "assumed_species": assumed_species,
                    "accuracy": acc,
                    "classification_report": report,
                },
                f,
                indent=2,
            )
        print("Saved metrics to:", out_path)


if __name__ == "__main__":
    main()
