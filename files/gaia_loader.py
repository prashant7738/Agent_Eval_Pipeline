"""
Pulls a stratified subset of text-only GAIA validation tasks (per roadmap.txt:
15-20 tasks instead of 90) and saves them to data/gaia_tasks.json.

Requires an HF_TOKEN with approved access to the gated gaia-benchmark/GAIA repo:
visit https://huggingface.co/datasets/gaia-benchmark/GAIA and click
"Agree and access repository" first, or this will raise GatedRepoError.
"""

import json
import os
import random
import re
from pathlib import Path

from dotenv import load_dotenv

DATA_DIR = Path(__file__).parent / "data"
OUT_PATH = DATA_DIR / "gaia_tasks.json"

# Excludes questions that need video/image understanding our text-only agent
# can't do, per the paper's inclusion criteria (sandbox has no vision/audio tool).
_MULTIMEDIA_PATTERN = re.compile(
    r"youtube\.com|youtu\.be|\bvideo\b|\bimage\b|\bphotograph\b|\.jpg|\.png", re.IGNORECASE
)


def load_gaia_validation_metadata():
    """Download the GAIA validation metadata.parquet and return it as a DataFrame."""
    import pandas as pd
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id="gaia-benchmark/GAIA",
        filename="2023/validation/metadata.parquet",
        repo_type="dataset",
        token=os.environ.get("HF_TOKEN"),
    )
    return pd.read_parquet(path)


def select_text_only_tasks(df, n_total: int = 18, seed: int = 42):
    """
    Filter to tasks with no attached file and no video/image dependency in the
    question text (text/web-only, per the paper's inclusion criteria), then
    sample proportionally across the three GAIA difficulty levels.
    """
    text_only = df[df["file_name"] == ""].copy()
    text_only["Level"] = text_only["Level"].astype(int)
    text_only = text_only[~text_only["Question"].str.contains(_MULTIMEDIA_PATTERN)]

    # Proportional split roughly matching the paper's 30/40/20% -> scaled to n_total.
    level_shares = {1: 0.35, 2: 0.45, 3: 0.20}
    rng = random.Random(seed)

    selected = []
    for level, share in level_shares.items():
        level_df = text_only[text_only["Level"] == level]
        n_level = max(1, round(n_total * share))
        rows = level_df.sample(n=min(n_level, len(level_df)), random_state=seed).to_dict("records")
        selected.extend(rows)

    rng.shuffle(selected)
    return selected[:n_total]


def build_task_set(n_total: int = 18, seed: int = 42):
    df = load_gaia_validation_metadata()
    rows = select_text_only_tasks(df, n_total=n_total, seed=seed)

    tasks = [
        {
            "task_id": row["task_id"],
            "level": int(row["Level"]),
            "question": row["Question"],
            "reference_answer": row["Final answer"],
        }
        for row in rows
    ]

    DATA_DIR.mkdir(exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(tasks, f, indent=2)

    return tasks


if __name__ == "__main__":
    load_dotenv(Path(__file__).parent / ".env")
    tasks = build_task_set(n_total=30)  # Change 18 to 30
    print(f"Saved {len(tasks)} tasks to {OUT_PATH}")
    for t in tasks:
        print(f"  [L{t['level']}] {t['task_id']}: {t['question'][:80]}")