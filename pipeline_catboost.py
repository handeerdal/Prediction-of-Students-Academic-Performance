# =============================================================================
# AI Assistance Disclosure
# -----------------------------------------------------------------------------
# Tool:    Claude (Sonnet 4.6, Anthropic)
# Date:    2025-05-12
# Prompt:  "Write a pipeline that calls my CatBoost training
#          function."
# Notes:   The generated code was reviewed and
#          partially adjusted by the author.
# =============================================================================
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from catboost_training import train_multiple_datasets

print("\n=== CatBoost Training ===")
datasets = [
    "60_ongoing_semester.csv", "50_ongoing_semester.csv",
    "40_ongoing_semester.csv", "30_ongoing_semester.csv",
    "20_ongoing_semester.csv", "next_semester_combined.csv",
    "nextyear.csv",
]
os.makedirs("results/model_files", exist_ok=True)
train_multiple_datasets(datasets, "data/prediction_datasets", "results/model_files", run_cv=True, cv_splits=5)

print("\n=== CATBOOST TRAINING COMPLETE ===")