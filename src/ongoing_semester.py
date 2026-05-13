import pandas as pd
import warnings
import os

warnings.filterwarnings("ignore")

from functions import (
    process_subject_predict_ongoing,
    prepare_base_data,
)

SUBJECT_LIST = [
    "Lithuanian Language",
    "Mathematics",
    "History",
    "Geography",
    "Biology",
    "English Language",
    "Physics",
    "Chemistry",
]
YEARS = [5,6,7,8,9,10,11,12]
DAYS = [20,30,40,50,60]


current_dir = os.getcwd()
input_dir = os.path.join(current_dir, "data")

input_file_students = os.path.join(input_dir, "students_preprocessed_sentetik.csv")

output_dir = os.path.join(current_dir, "data", "prediction_datasets")

os.makedirs(output_dir, exist_ok=True)

all_grades = pd.read_csv(input_file_students)
print("STUDENTS DATA IMPORTED")


def create_ongoingsemester_prediction_dataset(days, all_grades):
    students, modified_students, passed_df = prepare_base_data(all_grades)

    all_results = []

    for year in YEARS:
        for subject in SUBJECT_LIST:
            print(f"\nProcessing {subject} - Year {year}")
            try:
                available_semesters = modified_students[
                    (modified_students["ClassN"] == year)
                    & (modified_students["Subject"] == subject)
                ]["Semester_Num"].unique()
                available_semesters = sorted(
                    [s for s in available_semesters if s in [1, 2, 3]]
                )
                if not available_semesters:
                    print(f"No data available for {subject} - Year {year}")
                    continue
                for semester in available_semesters:
                    try:
                        result = process_subject_predict_ongoing(
                            students=students,
                            modified_students=modified_students,
                            passed_df=passed_df,
                            year=year,
                            subject=subject,
                            semester=semester,
                            days=days,
                        )
                        if not result.empty:
                            all_results.append(result)
                    except Exception as e:
                        print(
                            f"Error  {subject} - Year {year} - Semester {semester}: {str(e)}"
                        )
                        continue
            except Exception as e:
                print(f"Error  {subject} - Year {year}: {str(e)}")
                continue

    if all_results:
        final_universal_df = pd.concat(all_results, ignore_index=True)
        print(f"Total records: {len(final_universal_df)}")
        return final_universal_df
    else:
        print("No data processed.")
        return None

if __name__ == "__main__":
    for day in DAYS:
        print(f"Processing {day} days...")
        universal_df = create_ongoingsemester_prediction_dataset(day, all_grades)
        output_file = os.path.join(output_dir, f"{day}_ongoing_semester.csv")
        universal_df.to_csv(output_file, index=False)

        if universal_df is not None:
            print(f"Dataset shape: {universal_df.shape}")

        del universal_df
        print(f"Completed {day} days\n")