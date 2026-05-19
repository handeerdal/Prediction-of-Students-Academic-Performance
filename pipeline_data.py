# =============================================================================
# AI Assistance Disclosure
# -----------------------------------------------------------------------------
# Tool:    Claude (Sonnet 4.6, Anthropic)
# Date:    2025-05-12
# Prompt:  "Write a pipeline that runs my dataset creation scripts
#          in sequence."
# Notes:   The generated code was reviewed and partially adjusted
#          by the author.
# =============================================================================

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from ongoing_semester import create_ongoingsemester_prediction_dataset
from next_semester import create_nextsemester_prediction_dataset
from nextyear_firstsemr import create_nextyear_firstsemester_prediction_dataset
from nextyear import predictiondataset
from combine_semesters import concatsemesters

import pandas as pd

print("\n=== STEP 1: Ongoing Semester ===")
DAYS = [20, 30, 40, 50, 60]
all_grades = pd.read_csv("data/students_preprocessed_synthetic.csv")
os.makedirs("data/prediction_datasets", exist_ok=True)
for day in DAYS:
    print(f"Processing {day} days...")
    df = create_ongoingsemester_prediction_dataset(day, all_grades)
    if df is not None:
        df.to_csv(f"data/prediction_datasets/{day}_ongoing_semester.csv", index=False)
    del df

print("\n=== STEP 2a: Next Semester (Sem1 -> Sem2) ===")
create_nextsemester_prediction_dataset()

print("\n=== STEP 2b: Next Semester (year-end -> next year Sem1) ===")
create_nextyear_firstsemester_prediction_dataset()

print("\n=== STEP 3: Combine Next Semester ===")
data1 = pd.read_csv("data/prediction_datasets/SEM1TOSEM2.csv")
data2 = pd.read_csv("data/prediction_datasets/nextyear_1stsemester.csv")
combined = concatsemesters(data1, data2)
combined["Sem1_to_sem2"] = combined["Sem1_to_sem2"].fillna(0).astype(int)
combined["NextYear_Sem1"] = combined["NextYear_Sem1"].fillna(0).astype(int)
combined.to_csv("data/prediction_datasets/next_semester_combined.csv", index=False)
print(f"Saved: next_semester_combined.csv  shape={combined.shape}")

print("\n=== STEP 4: Next Year ===")
predictiondataset(years=[5, 6, 7, 8, 9, 10, 11], pred=1, name="nextyear")

print("\n=== DATA CREATION COMPLETE ===")