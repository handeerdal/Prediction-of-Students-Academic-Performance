import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings("ignore")

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


def prepare_base_data(all_grades):
    students = all_grades.copy()
    students["Date"] = pd.to_datetime(students["Date"])
    modified_students = students.copy()
    modified_students = modified_students[
        (modified_students["Passed"] == 0) & (modified_students["Not_Passed"] == 0)
    ]
    passed_df = students.copy()
    passed_df = passed_df[(passed_df["Passed"] > 0) | (passed_df["Not_Passed"] > 0)]
    return students, modified_students, passed_df


def check_and_remove_multi_classletter(data):
    if data.empty:
        return data
    letter_counts = data.groupby(["StudentID", "Subject", "Semester_Num", "ClassN"])[
        "ClassSection"
    ].nunique()
    multi_letter_mask = letter_counts > 1

    if multi_letter_mask.any():
        multi_classletter_stids = (
            letter_counts[multi_letter_mask].index.get_level_values("StudentID").unique()
        )
        data = data[~data["StudentID"].isin(multi_classletter_stids)]
        print(
            f"Found {len(multi_classletter_stids)} students with multiple ClassSections"
        )
    return data


def get_semester_data(modified_students, year, subject, semester_num):
    data = modified_students[
        (modified_students["ClassN"] == year)
        & (modified_students["Subject"] == subject)
        & (modified_students["Semester_Num"] == semester_num)
    ].copy()
    return data


def get_all_lectures_data(students, year, semester_num):
    if semester_num == 0:
        return students[(students["ClassN"] == year)].copy()
    elif year == 0:
        return students[(students["Semester_Num"] == semester_num)].copy()
    else:
        return students[
            (students["ClassN"] == year)
            & (students["Semester_Num"] == semester_num)
        ].copy()


def get_other_lectures_data(modified_students, year, subject, semester_num):
    if semester_num == 0:
        return modified_students[
            (modified_students["ClassN"] == year)
            & (modified_students["Subject"] != subject)
        ].copy()
    elif year == 0:
        return modified_students[
            (modified_students["Semester_Num"] == semester_num)
            & (modified_students["Subject"] != subject)
        ].copy()
    else:
        return modified_students[
            (modified_students["ClassN"] == year)
            & (modified_students["Subject"] != subject)
            & (modified_students["Semester_Num"] == semester_num)
        ].copy()


def apply_time_shift(data, latest_dates, days, scid_col="SchoolID", acyear_col="AcademicYear"):
    if data.empty:
        return data

    data = data.merge(
        latest_dates[["SchoolID", "AcademicYear", f"{days}_before"]],
        left_on=[scid_col, acyear_col],
        right_on=["SchoolID", "AcademicYear"],
        how="left",
    )
    data = data[data["Date"] <= data[f"{days}_before"]].copy()
    data.drop(columns=[f"{days}_before"], inplace=True)
    return data


def calculate_grades_improving(semester_data):
    if semester_data.empty:
        return pd.DataFrame()

    semester_data["Month"] = semester_data["Date"].dt.to_period("M").astype(str)
    results = []

    for (student_id, classnumber, classgroup_id), group in semester_data.groupby(
        ["StudentID", "ClassN", "ClassGroup"]
    ):
        group = group.sort_values(by="Month")
        monthly = group.groupby("Month")["Grade"].mean().dropna().reset_index()

        if len(monthly) < 2:
            continue

        monthly.columns = ["Month", "Avg"]
        monthly["Month_Index"] = range(1, len(monthly) + 1)

        cumulative_avgs = []
        for i in range(len(monthly)):
            current_month = monthly.loc[i, "Month"]
            grades_until_now = group[
                (group["Month"] <= current_month) & (group["Grade"].notna())
            ]["Grade"]
            cumulative_avg = grades_until_now.mean()
            cumulative_avgs.append(cumulative_avg)

        monthly_result = pd.DataFrame(
            {
                "Month_Index": monthly["Month_Index"],
                "Cumulative_Avg": cumulative_avgs,
                "StudentID": student_id,
                "ClassGroup": classgroup_id,
                "ClassN": classnumber,
            }
        )
        results.append(monthly_result)

    if not results:
        return pd.DataFrame()

    results = pd.concat(results, ignore_index=True)
    results = results.dropna(subset="Cumulative_Avg")

    regression_rows = []
    for (student_id, classnumber, classgroup_id), group in results.groupby(
        ["StudentID", "ClassN", "ClassGroup"]
    ):
        X = group[["Month_Index"]]
        y = group["Cumulative_Avg"]

        if len(group) >= 2:
            model = LinearRegression()
            model.fit(X, y)
            slope = model.coef_[0]
            intercept = model.intercept_
        else:
            slope = np.nan
            intercept = np.nan

        group = group.copy()
        group["Grade_Trend_Slope"] = slope
        group["Grade_Trend_Intercept"] = intercept
        regression_rows.append(group)

    if not regression_rows:
        return pd.DataFrame()

    results = pd.concat(regression_rows, ignore_index=True)
    results_reduced = results.drop_duplicates(
        subset=["StudentID", "ClassN", "ClassGroup"]
    )

    return results_reduced[
        ["StudentID", "ClassN", "ClassGroup", "Grade_Trend_Slope", "Grade_Trend_Intercept"]
    ]


def select_best_classgroup_for_subjects_predict_next_year(data, target_subjects):
    target_data = data[data["Subject"].isin(target_subjects)].copy()

    if target_data.empty:
        return data

    grade_counts = (
        target_data.groupby(["StudentID", "Subject", "ClassN", "ClassGroup"])["Grade"]
        .count()
        .reset_index(name="grade_count")
    )

    best_groups = grade_counts.loc[
        grade_counts.groupby(["StudentID", "Subject", "ClassN"])["grade_count"].idxmax()
    ][["StudentID", "Subject", "ClassN", "ClassGroup"]]

    filtered_target_data = target_data.merge(
        best_groups, on=["StudentID", "Subject", "ClassN", "ClassGroup"], how="inner"
    )

    non_target_data = data[~data["Subject"].isin(target_subjects)].copy()
    result = pd.concat([filtered_target_data, non_target_data], ignore_index=True)
    return result


def select_best_classgroup_for_subjects(data, target_subjects):
    target_data = data[data["Subject"].isin(target_subjects)].copy()

    if target_data.empty:
        return data

    classgroup_counts = (
        target_data.groupby(["StudentID", "Subject", "Semester_Num", "ClassN"])[
            "ClassGroup"
        ]
        .nunique()
        .reset_index(name="num_classgroups")
    )

    multiple_classgroups = classgroup_counts[classgroup_counts["num_classgroups"] > 1]

    if multiple_classgroups.empty:
        return data

    problematic_data = target_data.merge(
        multiple_classgroups[["StudentID", "Subject", "Semester_Num", "ClassN"]],
        on=["StudentID", "Subject", "Semester_Num", "ClassN"],
        how="inner",
    )

    grade_counts = (
        problematic_data.groupby(
            ["StudentID", "Subject", "Semester_Num", "ClassN", "ClassGroup"]
        )["Grade"]
        .count()
        .reset_index(name="grade_count")
    )

    best_groups = grade_counts.loc[
        grade_counts.groupby(["StudentID", "Subject", "Semester_Num", "ClassN"])[
            "grade_count"
        ].idxmax()
    ][["StudentID", "Subject", "Semester_Num", "ClassN", "ClassGroup"]]

    filtered_problematic_data = problematic_data.merge(
        best_groups,
        on=["StudentID", "Subject", "Semester_Num", "ClassN", "ClassGroup"],
        how="inner",
    )

    clean_data = target_data.merge(
        classgroup_counts[classgroup_counts["num_classgroups"] == 1][
            ["StudentID", "Subject", "Semester_Num", "ClassN"]
        ],
        on=["StudentID", "Subject", "Semester_Num", "ClassN"],
        how="inner",
    )
    non_target_data = data[~data["Subject"].isin(target_subjects)].copy()

    result = pd.concat(
        [filtered_problematic_data, clean_data, non_target_data], ignore_index=True
    )
    return result


def run_all_steps(
    students,
    modified_students,
    year,
    semester,
    latest_dates,
    days,
    subject,
    prev_data,
    passed_df,
    finaldf,
):
    all_lectures_data = get_all_lectures_data(students, year, semester)
    other_lectures_data = get_other_lectures_data(modified_students, year, subject, semester)

    if not other_lectures_data.empty:
        if semester == 0:
            pass
        else:
            other_lectures_data = select_best_classgroup_for_subjects(
                other_lectures_data, SUBJECT_LIST
            )

    if not all_lectures_data.empty:
        if semester == 0:
            pass
        else:
            all_lectures_data = select_best_classgroup_for_subjects(
                all_lectures_data, SUBJECT_LIST
            )

    if latest_dates is not None:
        all_lectures_data = apply_time_shift(all_lectures_data, latest_dates, days)
        other_lectures_data = apply_time_shift(other_lectures_data, latest_dates, days)

    if not other_lectures_data.empty:
        others_mean = (
            other_lectures_data.groupby(["StudentID", "ClassN"])["Grade"]
            .agg("mean")
            .reset_index()
        )
        finaldf = finaldf.merge(others_mean, on=["StudentID", "ClassN"], how="left")
        finaldf = finaldf.rename(columns={"Grade": "GPA_OtherSubjects"})

    if not all_lectures_data.empty:
        subject_counts = (
            all_lectures_data.groupby(["StudentID", "ClassN"])["Subject"]
            .nunique()
            .reset_index()
        )
        subject_counts.rename(columns={"Subject": "Unique_Subject_Count"}, inplace=True)
        finaldf = finaldf.merge(subject_counts, on=["StudentID", "ClassN"], how="left")

    if not all_lectures_data.empty:
        attendance = all_lectures_data[all_lectures_data["Subject"] == subject]
        subjattendance = attendance.groupby(
            ["StudentID", "ClassN", "ClassGroup"], as_index=False
        )["Absence"].sum()
        finaldf = finaldf.merge(
            subjattendance, on=["StudentID", "ClassN", "ClassGroup"], how="left"
        )
        finaldf = finaldf.rename(columns={"Absence": "Absences_Target"})
        finaldf["Absences_Target"] = finaldf["Absences_Target"].fillna(0).astype(int)

    if not all_lectures_data.empty:
        attendanceother = all_lectures_data[all_lectures_data["Subject"] != subject]
        attendance_other = attendanceother.groupby(
            ["StudentID", "ClassN"], as_index=False
        )["Absence"].sum()
        finaldf = finaldf.merge(attendance_other, on=["StudentID", "ClassN"], how="left")
        finaldf = finaldf.rename(columns={"Absence": "Absences_OtherSubjects"})
        finaldf["Absences_OtherSubjects"] = finaldf["Absences_OtherSubjects"].fillna(0).astype(int)

    if not prev_data.empty:
        student_std = (
            prev_data.groupby(["StudentID", "ClassN", "ClassGroup"])["Grade"]
            .std()
            .reset_index()
        )
        student_std.rename(columns={"Grade": "Grade_SD_Target"}, inplace=True)
        finaldf = finaldf.merge(student_std, on=["StudentID", "ClassN", "ClassGroup"], how="left")

        studentmin = (
            prev_data.groupby(["StudentID", "ClassN", "ClassGroup"])["Grade"]
            .min()
            .reset_index()
        )
        studentmin.rename(columns={"Grade": "Min_Grade_Target"}, inplace=True)
        finaldf = finaldf.merge(studentmin, on=["StudentID", "ClassN", "ClassGroup"], how="left")

        studentmax = (
            prev_data.groupby(["StudentID", "ClassN", "ClassGroup"])["Grade"]
            .max()
            .reset_index()
        )
        studentmax.rename(columns={"Grade": "Max_Grade_Target"}, inplace=True)
        finaldf = finaldf.merge(studentmax, on=["StudentID", "ClassN", "ClassGroup"], how="left")

        student_medians = (
            prev_data.groupby(["StudentID", "ClassN", "ClassGroup"])["Grade"]
            .median()
            .reset_index()
        )
        student_medians.rename(columns={"Grade": "Median_Grade_Target"}, inplace=True)
        finaldf = finaldf.merge(student_medians, on=["StudentID", "ClassN", "ClassGroup"], how="left")

        def min_mode(series):
            modes = series.dropna().mode()
            if modes.empty:
                return np.nan
            return modes.min()

        def max_mode(series):
            modes = series.dropna().mode()
            if modes.empty:
                return np.nan
            return modes.max()

        mod_df = (
            prev_data.groupby(["StudentID", "ClassN", "ClassGroup"])["Grade"]
            .agg(**{"Mode_MinGrade_Target": min_mode, "Mode_MaxGrade_Target": max_mode})
            .reset_index()
        )
        finaldf = finaldf.merge(mod_df, on=["StudentID", "ClassN", "ClassGroup"], how="left")

        prev_data_nonzero = prev_data[prev_data["Grade"] != 0].copy()
        count_sem = (
            prev_data_nonzero.groupby(["StudentID", "ClassN", "ClassGroup"])["Grade"]
            .count()
            .reset_index()
        )
        count_sem.rename(columns={"Grade": "Assessment_Count_Target"}, inplace=True)
        finaldf = finaldf.merge(count_sem, on=["StudentID", "ClassN", "ClassGroup"], how="left")

        class_medians = (
            prev_data.groupby(["AcademicYear", "ClassSection", "ClassN", "SchoolID"])["Grade"]
            .median()
            .reset_index(name="ClassSection_Median")
        )
        finaldf = finaldf.merge(
            class_medians, on=["AcademicYear", "ClassN", "ClassSection", "SchoolID"], how="left"
        )

        class_mean = (
            prev_data.groupby(["AcademicYear", "ClassSection", "ClassN", "SchoolID"])["Grade"]
            .mean()
            .reset_index(name="ClassSection_Mean")
        )
        finaldf = finaldf.merge(
            class_mean, on=["AcademicYear", "ClassN", "ClassSection", "SchoolID"], how="left"
        )

        groupmedian = (
            prev_data.groupby(["AcademicYear", "ClassN", "ClassGroup"])["Grade"]
            .median()
            .reset_index(name="ClassGroup_Median")
        )
        finaldf = finaldf.merge(groupmedian, on=["AcademicYear", "ClassN", "ClassGroup"], how="left")

        groupmean = (
            prev_data.groupby(["AcademicYear", "ClassN", "ClassGroup"])["Grade"]
            .mean()
            .reset_index(name="ClassGroup_Mean")
        )
        finaldf = finaldf.merge(groupmean, on=["AcademicYear", "ClassN", "ClassGroup"], how="left")

        improving_results = calculate_grades_improving(prev_data)
        if not improving_results.empty:
            finaldf = finaldf.merge(
                improving_results[["StudentID", "ClassGroup", "Grade_Trend_Slope", "Grade_Trend_Intercept"]],
                on=["StudentID", "ClassGroup"],
                how="left",
            )

    subjnotincluded = other_lectures_data[other_lectures_data["Grade"] != 0]
    if not subjnotincluded.empty:
        count_exsubj = (
            subjnotincluded.groupby(["StudentID", "ClassN"])["Grade"]
            .count()
            .reset_index()
        )
        count_exsubj.rename(columns={"Grade": "Assessment_Count_Others"}, inplace=True)
        finaldf = finaldf.merge(count_exsubj, on=["StudentID", "ClassN"], how="left")

    if (semester != 0) & (year != 0):
        passed_semester_data = passed_df[
            (passed_df["ClassN"] == year) & (passed_df["Semester_Num"] == semester)
        ].copy()
    elif (semester != 0) & (year == 0):
        passed_semester_data = passed_df[(passed_df["Semester_Num"] == semester)].copy()
    elif (semester == 0) & (year != 0):
        passed_semester_data = passed_df[(passed_df["ClassN"] == year)].copy()

    if not passed_semester_data.empty:
        if semester == 0:
            pass
        else:
            passed_semester_data = select_best_classgroup_for_subjects(
                passed_semester_data, SUBJECT_LIST
            )
        if latest_dates is not None:
            passed_semester_data = apply_time_shift(
                passed_semester_data, latest_dates, days
            )

        grouped_counts = (
            passed_semester_data.groupby(["StudentID", "ClassN", "SchoolID"])
            .agg(
                {
                    "Passed": lambda x: (x == 1).sum() if not x.isna().all() else np.nan,
                    "Not_Passed": lambda x: (x == 1).sum() if not x.isna().all() else np.nan,
                }
            )
            .reset_index()
        )

        grouped_counts.columns = [
            "StudentID", "ClassN", "SchoolID",
            "Total_Passed_Assessments", "Total_Failed_Assessments"
        ]
        merged_counts = grouped_counts[[
            "StudentID", "ClassN", "SchoolID",
            "Total_Passed_Assessments", "Total_Failed_Assessments"
        ]]
        finaldf = pd.merge(finaldf, merged_counts, on=["StudentID", "ClassN", "SchoolID"], how="left")

    if "Assessment_Count_Target" in finaldf.columns:
        finaldf = finaldf[finaldf["Assessment_Count_Target"] >= 2]

    if (year != 0):
        print(f"Completed {subject}  - Semester {semester}: {len(finaldf)} records")
    if (year == 0):
        print(f"Completed {subject} - Year {year} - Semester {semester}: {len(finaldf)} records")

    return finaldf


def process_subject_predict_ongoing(
    students,
    modified_students,
    passed_df,
    year,
    subject,
    semester,
    days,
):
    print(f"Processing {subject} - Year {year} - Semester {semester}")

    current_sem_data = get_semester_data(modified_students, year, subject, semester)

    if current_sem_data.empty:
        print(f"No data for {subject} - Year {year} - Semester {semester}")
        return pd.DataFrame()

    if subject in SUBJECT_LIST:
        current_sem_data = select_best_classgroup_for_subjects(current_sem_data, [subject])
        if current_sem_data.empty:
            return pd.DataFrame()

    current_sem_data = check_and_remove_multi_classletter(current_sem_data)

    if current_sem_data.empty:
        print(f"No data after ClassSection filtering for {subject} - Year {year}")
        return pd.DataFrame()

    latest_dates = (
        current_sem_data.groupby(["SchoolID", "AcademicYear"])["Date"].max().reset_index()
    )
    latest_dates["Date"] = pd.to_datetime(latest_dates["Date"])
    latest_dates[f"{days}_before"] = latest_dates["Date"] - pd.Timedelta(days=days)

    prev_data = apply_time_shift(current_sem_data.copy(), latest_dates, days)

    if not prev_data.empty:
        prevmonthmean = (
            prev_data.groupby(["StudentID", "ClassGroup"])["Grade"]
            .agg("mean")
            .reset_index()
        )
        prevmonthmean = prevmonthmean.rename(columns={"Grade": "Prior_GPA_Subject"})

        finaldf = (
            prev_data.groupby(
                ["ClassN", "ClassSection", "ClassGroup", "StudentID", "SchoolID",
                 "AcademicYear", "TeacherID", "Subject", "Semester"]
            )
            .first()
            .reset_index()[
                ["ClassN", "ClassSection", "ClassGroup", "StudentID", "SchoolID",
                 "AcademicYear", "TeacherID", "Subject", "Semester"]
            ]
        )

        finaldf = finaldf.merge(prevmonthmean, on=["StudentID", "ClassGroup"], how="left")

    target_grades = (
        current_sem_data.groupby(
            ["ClassN", "ClassSection", "ClassGroup", "StudentID", "SchoolID",
             "AcademicYear", "TeacherID", "Subject", "Semester"]
        )["Grade"]
        .agg(["mean"])
        .reset_index()
    )
    target_grades = target_grades.rename(columns={"mean": "Target_GPA"})

    finaldf = finaldf.merge(
        target_grades[["StudentID", "ClassGroup", "Target_GPA"]],
        on=["StudentID", "ClassGroup"],
        how="left",
    )

    key_columns = ["StudentID", "Subject", "AcademicYear"]
    duplicated_records = finaldf.duplicated(subset=key_columns, keep=False)
    nan_grades = finaldf["Target_GPA"].isna()
    finaldf = finaldf[~(duplicated_records & nan_grades)]

    finaldf = run_all_steps(
        students=students,
        modified_students=modified_students,
        year=year,
        semester=semester,
        latest_dates=latest_dates,
        days=days,
        subject=subject,
        prev_data=prev_data,
        passed_df=passed_df,
        finaldf=finaldf,
    )

    return finaldf