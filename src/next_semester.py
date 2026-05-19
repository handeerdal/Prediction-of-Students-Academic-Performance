# =============================================================================
# AI Assistance Disclosure
# -----------------------------------------------------------------------------
# Tool:    Claude (Sonnet 4.6, Anthropic)
# Date:    2025-04-15
# Prompt:  "Using the functions in function.py, write the call lines that run
#          them for each subject and combine the results into the next-semester
#          dataset as a .py file instead of a Jupyter notebook."
# Notes:   The data processing functions were written by the author in a
#          Jupyter notebook and converted into a single .py file with the help
#          of AI. In this file, AI was used only to add the call layer that
#          invokes these functions in a loop over subjects. The generated code
#          was reviewed and partially adjusted by the author.
# =============================================================================

import pandas as pd
import os

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

from functions import (
    select_best_classgroup_for_subjects,
    check_and_remove_multi_classletter,
    prepare_base_data,
    run_all_steps,
)


def prepare_subject_data(modified_students, subject, semester):
    print(f"Processing: {subject}, Semester: {semester}")
    modified_students = select_best_classgroup_for_subjects(
        modified_students, SUBJECT_LIST
    )
    subject_data = modified_students[
        (modified_students["Subject"] == subject) & 
        (modified_students["Semester_Num"] == semester)
    ]
    subject_data = check_and_remove_multi_classletter(subject_data)
    semester_data = subject_data[subject_data["Semester_Num"] == semester].copy()

    if len(semester_data) == 0:
        return None
    return semester_data


def process_subject(students, subject, subject_list, sem_from, sem_to):

    students, modified_students, passed_df = prepare_base_data(students)

    prev_sem_data = prepare_subject_data(modified_students, subject, sem_from)
    target_sem_data = prepare_subject_data(modified_students, subject, sem_to)

    if prev_sem_data is None:
        print(f"NAN data for {subject} (Sem {sem_from}→{sem_to})")
        return None

    finaldf = (
        prev_sem_data.groupby(
            [
                "ClassN",
                "ClassSection",
                "ClassGroup",
                "StudentID",
                "SchoolID",
                "AcademicYear",
                "TeacherID",
                "Subject",
                "Semester",
            ]
        )["Grade"]
        .agg(["mean"])
        .reset_index()
    )

    finaldf = finaldf.rename(columns={"mean": "Prior_GPA_Subject"})

    second_sem_target = (
        target_sem_data.groupby(["StudentID", "ClassN"])["Grade"]
        .agg(["mean"])
        .reset_index()
    )
    finaldf = finaldf.merge(second_sem_target, on=["StudentID", "ClassN"], how="left")
    finaldf = finaldf.rename(columns={"mean": "Target_GPA"})

    key_columns = ["StudentID", "Subject", "AcademicYear", "ClassN"]
    duplicated_records = finaldf.duplicated(subset=key_columns, keep=False)
    nan_grades = finaldf["Target_GPA"].isna()

    finaldf = finaldf[~(duplicated_records & nan_grades)]
    finaldf = finaldf.dropna(subset=["Prior_GPA_Subject"])

    finaldf = run_all_steps(
        students,
        modified_students,
        year=0,
        semester=sem_from,
        latest_dates=None,
        days=0,
        subject=subject,
        prev_data=prev_sem_data,
        passed_df=passed_df,
        finaldf=finaldf,
    )
    return finaldf


def create_nextsemester_prediction_dataset():

    current_dir = os.getcwd()
    input_dir = os.path.join(current_dir, "data")
    input_file_students = os.path.join(input_dir, "students_preprocessed_synthetic.csv")
    output_dir = os.path.join(current_dir, "data", "prediction_datasets")

    os.makedirs(output_dir, exist_ok=True)

    students = pd.read_csv(input_file_students)
    print("STUDENTS DATA IMPORTED")

    semester_pairs = [
        (1, 2, "SEM1TOSEM2")
     
    ]
    for sem_from, sem_to, output_name in semester_pairs:
        all_dataframes = []
        print(f"Processing Semester {sem_from} --> Semester {sem_to}")

        for subject in SUBJECT_LIST:
            result_df = process_subject(
                students, subject, SUBJECT_LIST, sem_from, sem_to
            )

            if result_df is not None:
                all_dataframes.append(result_df)
            else:
                print('ERROR! NAN DATASET')

        if len(all_dataframes) > 0:
            final_combined_df = pd.concat(all_dataframes, ignore_index=True)
            output_file = os.path.join(output_dir, f"{output_name}.csv")
            final_combined_df.to_csv(output_file, index=False)
            print(f"\nCombined file saved: {output_file}")
        else:
            print("\nNo data to combine!")


if __name__ == "__main__":
    create_nextsemester_prediction_dataset()