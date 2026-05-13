import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from basemodels_training import train_multiple_datasets

print("\n=== Base Model Comparison ===")
datasets = [
    "20_ongoing_semester.csv", "30_ongoing_semester.csv",
    "40_ongoing_semester.csv", "50_ongoing_semester.csv",
    "60_ongoing_semester.csv", "next_semester_combined.csv",
    "nextyear.csv",
]
os.makedirs("results/base_training_results", exist_ok=True)
train_multiple_datasets(datasets, "data/prediction_datasets", "results/base_training_results")

print("\n=== BASE MODEL TRAINING COMPLETE ===")