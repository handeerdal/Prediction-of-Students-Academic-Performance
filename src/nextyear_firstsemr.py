import pandas as pd
import numpy as np
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
YEARS = [5,6,7,8,9,10,11]
FILENAME = 'nextyear_1stsemester'

from functions import (
    select_best_classgroup_for_subjects,
    check_and_remove_multi_classletter,
    prepare_base_data,
    run_all_steps,
)


def process_subject_year(SUBJECT, CLASS_NUMBER, students, modified_students, passed_df):
    print(f"Processing {SUBJECT} for Class {CLASS_NUMBER}")
    source_year_data = modified_students[modified_students['ClassN'] == CLASS_NUMBER]
    passedincluded = students[students['ClassN'] == CLASS_NUMBER]
    source_year_data = select_best_classgroup_for_subjects(source_year_data, SUBJECT_LIST)

    if len(source_year_data) == 0:
        print(f"Skipped {SUBJECT} Class {CLASS_NUMBER}: no source data")
        return None

    last_semesters = source_year_data.groupby('StudentID')['Semester_Num'].max().reset_index()
    source_semester_data = source_year_data.merge(last_semesters, on=['StudentID', 'Semester_Num'])
    passedincluded = passedincluded.merge(last_semesters, on=['StudentID', 'Semester_Num'])

    subject_last_semester = source_semester_data[source_semester_data['Subject'] == SUBJECT]

    if len(subject_last_semester) == 0:
        print(f"Skipped {SUBJECT} Class {CLASS_NUMBER}: no data in last semester")
        return None

    subject_last_semester = check_and_remove_multi_classletter(subject_last_semester)

    finaldf = subject_last_semester.groupby(
        ['ClassN', 'ClassSection', 'Semester', 'ClassGroup', 'StudentID', 'SchoolID', 'AcademicYear', 'TeacherID', 'Subject']
    )['Grade'].agg(['mean']).reset_index()
    finaldf = finaldf.rename(columns={"mean": "Prior_GPA_Subject"})

    key_columns = ['StudentID', 'Subject', 'AcademicYear', 'ClassN']
    duplicated_records = finaldf.duplicated(subset=key_columns, keep=False)
    nan_grades = finaldf['Prior_GPA_Subject'].isna()
    finaldf = finaldf[~(duplicated_records & nan_grades)]
    finaldf = finaldf.dropna()

    finaldf = run_all_steps(
        students=students,
        modified_students=modified_students,
        year=CLASS_NUMBER,
        semester=0,
        latest_dates=None,
        days=0,
        subject=SUBJECT,
        prev_data=subject_last_semester,
        passed_df=passed_df,
        finaldf=finaldf,
    )

    finaldf = finaldf.drop(columns=[
        'GPA_OtherSubjects',
        'Unique_Subject_Count',
        'Absences_Target',
        'Absences_OtherSubjects',
        'Total_Passed_Assessments',
        'Total_Failed_Assessments'
    ], errors='ignore')

    other_subjects_last = source_semester_data[source_semester_data['Subject'] != SUBJECT]
    others_mean = other_subjects_last.groupby(['StudentID', 'ClassN'])['Grade'].agg('mean').reset_index()
    finaldf = finaldf.merge(others_mean, on=['StudentID', 'ClassN'], how='left')
    finaldf = finaldf.rename(columns={"Grade": "GPA_OtherSubjects"})

    subject_counts = passedincluded.groupby(['StudentID', 'ClassN'])['Subject'].nunique().reset_index()
    subject_counts.rename(columns={'Subject': 'Unique_Subject_Count'}, inplace=True)
    finaldf = finaldf.merge(subject_counts, on=['StudentID', 'ClassN'], how='left')

    attendanceother = passedincluded[passedincluded['Subject'] != SUBJECT]

    attendance_subj = subject_last_semester.groupby(['ClassN', 'StudentID'], as_index=False)['Absence'].sum()
    finaldf = finaldf.merge(attendance_subj, on=['ClassN', 'StudentID'], how="left")
    finaldf = finaldf.rename(columns={"Absence": "Absences_Target"})

    attendance_other = attendanceother.groupby(['ClassN', 'StudentID'], as_index=False)['Absence'].sum()
    finaldf = finaldf.merge(attendance_other, on=['ClassN', 'StudentID'], how="left")
    finaldf = finaldf.rename(columns={"Absence": "Absences_OtherSubjects"})

    last_sem_passed = passed_df[passed_df['ClassN'] == CLASS_NUMBER].copy()

    if len(last_sem_passed) > 0:
        last_semester_passed = last_sem_passed.groupby('StudentID')['Semester_Num'].max().reset_index()
        last_sem_passed = last_sem_passed.merge(last_semester_passed, on=['StudentID', 'Semester_Num'])

        grouped_counts = last_sem_passed.groupby(['StudentID', 'ClassN', 'SchoolID']).agg({
            'Passed': lambda x: (x == 1).sum() if not x.isna().all() else np.nan,
            'Not_Passed': lambda x: (x == 1).sum() if not x.isna().all() else np.nan
        }).reset_index()

        grouped_counts.columns = ['StudentID', 'ClassN', 'SchoolID', 'Total_Passed_Assessments', 'Total_Failed_Assessments']
        merged_counts = grouped_counts[['StudentID', 'Total_Passed_Assessments', 'ClassN', 'Total_Failed_Assessments']]
        finaldf = pd.merge(finaldf, merged_counts, on=['StudentID', 'ClassN'], how='left')

    finaldf['Total_Passed_Assessments'] = finaldf.get('Total_Passed_Assessments', 0).fillna(0)
    finaldf['Total_Failed_Assessments'] = finaldf.get('Total_Failed_Assessments', 0).fillna(0)

    target_students = students[
        (students['Semester_Num'] == 1) &
        (students['Subject'] == SUBJECT) &
        (students['ClassN'] == CLASS_NUMBER + 1)
    ]
    target_students = select_best_classgroup_for_subjects(target_students, SUBJECT_LIST)
    target_students = check_and_remove_multi_classletter(target_students)

    if len(target_students) == 0:
        print(f"Skipped {SUBJECT} Class {CLASS_NUMBER}: no target data for Class {CLASS_NUMBER + 1}")
        return None

    target_df = target_students.groupby(
        ['ClassN', 'ClassSection', 'ClassGroup', 'StudentID', 'SchoolID', 'AcademicYear', 'TeacherID', 'Subject']
    )['Grade'].agg(['mean']).reset_index()
    target_df = target_df.rename(columns={"mean": "Target_GPA"})
    target_df = target_df[['Target_GPA', 'StudentID', 'ClassN']]

    target_df = target_df[
        (target_df['Target_GPA'].notna()) &
        (target_df['Target_GPA'] > 0)
    ]

    finaldf['ClassN'] = CLASS_NUMBER + 1

    merged_df = target_df.merge(finaldf, on=['StudentID', 'ClassN'], how='left')
    merged_df = merged_df.drop_duplicates(keep='first')
    merged_df = merged_df.loc[:, ~merged_df.columns.str.endswith('_y')]
    merged_df.columns = merged_df.columns.str.replace('_x', '', regex=False)
    merged_df = merged_df.dropna(subset=['Target_GPA'])
    merged_df = merged_df.dropna(subset=['Prior_GPA_Subject'])

    print(f"Final shape: {merged_df.shape}")
    return merged_df


def create_nextyear_firstsemester_prediction_dataset():
    current_dir = os.getcwd()
    input_dir = os.path.join(current_dir, "data")
    input_file_students = os.path.join(input_dir, "students_preprocessed_synthetic.csv")

    students = pd.read_csv(input_file_students)
    print("STUDENTS DATA IMPORTED")

    students, modified_students, passed_df = prepare_base_data(students)

    all_results = []
    for subject in SUBJECT_LIST:
        for year in YEARS:
            try:
                result = process_subject_year(
                    subject, year,
                    students, modified_students, passed_df,
                )
                if result is not None and len(result) > 0:
                    all_results.append(result)
            except Exception as e:
                print(f"Error processing {subject} Class {year}: {str(e)}")
                continue

    if all_results:
        current_dir = os.getcwd()
        combined_df = pd.concat(all_results, ignore_index=True)
        output_dir = os.path.join(current_dir, "data", "prediction_datasets")
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{FILENAME}.csv")
        combined_df.to_csv(output_file, index=False)
        print(f"Combined dataset saved: {combined_df.shape}")

if __name__ == "__main__":
    create_nextyear_firstsemester_prediction_dataset()