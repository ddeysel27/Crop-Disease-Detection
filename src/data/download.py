"""Placeholders for dataset download helpers.
In practice, many leaf datasets require manual acceptance (Kaggle/Mendeley).
Use this file for any automated mirrors you have permission to fetch.
"""
from pathlib import Path

def note():
    return "Put downloaded archives under data/raw/, then run build_dataset.py"
