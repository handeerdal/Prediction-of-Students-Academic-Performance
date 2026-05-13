import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
import warnings
warnings.filterwarnings("ignore")
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


def load_all_data():
    current_dir = os.getcwd()
    input_dir = os.path.join(current_dir, "data")
    input_file_students = os.path.join(input_dir, "students_preprocessed_sentetik.csv")
    output_dir = os.path.join(current_dir, "data", "prediction_datasets")
    os.makedirs(output_dir, exist_ok=True)

    students = pd.read_csv(input_file_students)
    print("STUDENTS DATA IMPORTED")
    return students, output_dir


def predictiondataset(years, pred, name):
    students, output_dir = load_all_data()
    students["Date"] = pd.to_datetime(students["Date"])
    all_subjects = []

    for subject in SUBJECT_LIST:
        subject_df = process_subject(subject, students, years, pred)
        all_subjects.append(subject_df)

    final_df = pd.concat(all_subjects, ignore_index=True)
    output_file = os.path.join(output_dir, f"{name}.csv")
    final_df.to_csv(output_file, index=False)
    print(f"Combined dataset saved: {final_df.shape}")


def process_subject(subject, students, years, pred):
    all_years_subject = []
    for year in years:
        print(f"Processing {subject} - Year {year}")

        modified_students = students.copy()
        modified_students = modified_students[
            (modified_students["Passed"] == 0) & (modified_students["Not_Passed"] == 0)
        ]
        passed_df = students.copy()
        passed_df = passed_df[(passed_df["Passed"] > 0) | (passed_df["Not_Passed"] > 0)]

        current_class_data = modified_students[modified_students["ClassN"] == year]
        subject_current_year = current_class_data[current_class_data["Subject"] == subject]

        finaldf = (
            subject_current_year.groupby(
                ["ClassN", "StudentID", "SchoolID", "TeacherID", "Subject"]
            )["Grade"]
            .agg(["mean"])
            .reset_index()
        )
        finaldf = finaldf.rename(columns={"mean": "Prior_GPA_Subject"})

        all_lectures_data = students[(students["ClassN"] == year)].copy()
        other_lectures_data = modified_students[
            (modified_students["ClassN"] == year)
            & (modified_students["Subject"] != subject)
        ].copy()

        others_mean = (
            other_lectures_data.groupby(["StudentID", "ClassN"])["Grade"]
            .agg("mean")
            .reset_index()
        )
        finaldf = finaldf.merge(others_mean, on=["StudentID", "ClassN"], how="left")
        finaldf = finaldf.rename(columns={"Grade": "GPA_OtherSubjects"})

        subject_counts = (
            all_lectures_data.groupby(["StudentID", "ClassN"])["Subject"]
            .nunique()
            .reset_index()
        )
        subject_counts.rename(columns={"Subject": "Unique_Subject_Count"}, inplace=True)
        finaldf = finaldf.merge(subject_counts, on=["StudentID", "ClassN"], how="left")

        attendance = all_lectures_data[all_lectures_data["Subject"] == subject]
        subjattendance = attendance.groupby(
            ["StudentID", "ClassN"], as_index=False
        )["Absence"].sum()
        finaldf = finaldf.merge(subjattendance, on=["StudentID", "ClassN"], how="left")
        finaldf = finaldf.rename(columns={"Absence": "Absences_Target"})
        finaldf["Absences_Target"] = finaldf["Absences_Target"].fillna(0).astype(int)

        attendanceother = all_lectures_data[all_lectures_data["Subject"] != subject]
        attendance_other = attendanceother.groupby(
            ["StudentID", "ClassN"], as_index=False
        )["Absence"].sum()
        finaldf = finaldf.merge(attendance_other, on=["StudentID", "ClassN"], how="left")
        finaldf = finaldf.rename(columns={"Absence": "Absences_OtherSubjects"})
        finaldf["Absences_OtherSubjects"] = finaldf["Absences_OtherSubjects"].fillna(0).astype(int)

        student_std = (
            subject_current_year.groupby(["StudentID", "ClassN"])["Grade"]
            .std()
            .reset_index()
        )
        student_std.rename(columns={"Grade": "Grade_SD_Target"}, inplace=True)
        finaldf = finaldf.merge(student_std, on=["StudentID", "ClassN"], how="left")

        studentmin = (
            subject_current_year.groupby(["StudentID", "ClassN"])["Grade"]
            .min()
            .reset_index()
        )
        studentmin.rename(columns={"Grade": "Min_Grade_Target"}, inplace=True)
        finaldf = finaldf.merge(studentmin, on=["StudentID", "ClassN"], how="left")

        studentmax = (
            subject_current_year.groupby(["StudentID", "ClassN"])["Grade"]
            .max()
            .reset_index()
        )
        studentmax.rename(columns={"Grade": "Max_Grade_Target"}, inplace=True)
        finaldf = finaldf.merge(studentmax, on=["StudentID", "ClassN"], how="left")

        student_medians = (
            subject_current_year.groupby(["StudentID", "ClassN"])["Grade"]
            .median()
            .reset_index()
        )
        student_medians.rename(columns={"Grade": "Median_Grade_Target"}, inplace=True)
        finaldf = finaldf.merge(student_medians, on=["StudentID", "ClassN"], how="left")

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
            subject_current_year.groupby(["StudentID", "ClassN"])["Grade"]
            .agg(**{"Mode_MinGrade_Target": min_mode, "Mode_MaxGrade_Target": max_mode})
            .reset_index()
        )
        finaldf = finaldf.merge(mod_df, on=["StudentID", "ClassN"], how="left")

        prev_data_nonzero = subject_current_year[subject_current_year["Grade"] != 0].copy()
        count_sem = (
            prev_data_nonzero.groupby(["StudentID", "ClassN"])["Grade"]
            .count()
            .reset_index()
        )
        count_sem.rename(columns={"Grade": "Assessment_Count_Target"}, inplace=True)
        finaldf = finaldf.merge(count_sem, on=["StudentID", "ClassN"], how="left")

        def calculate_grades_improving(semester_data):
            if semester_data.empty:
                return pd.DataFrame()

            semester_data["Month"] = semester_data["Date"].dt.to_period("M").astype(str)
            results = []

            for (student_id, classnumber), group in semester_data.groupby(
                ["StudentID", "ClassN"]
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
                        "ClassN": classnumber,
                    }
                )
                results.append(monthly_result)

            if not results:
                return pd.DataFrame()

            results = pd.concat(results, ignore_index=True)
            results = results.dropna(subset="Cumulative_Avg")

            regression_rows = []
            for (student_id, classnumber), group in results.groupby(["StudentID", "ClassN"]):
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
            results_reduced = results.drop_duplicates(subset=["StudentID", "ClassN"])

            return results_reduced[["StudentID", "ClassN", "Grade_Trend_Slope", "Grade_Trend_Intercept"]]

        improving_results = calculate_grades_improving(subject_current_year)
        if not improving_results.empty:
            finaldf = finaldf.merge(
                improving_results[["StudentID", "Grade_Trend_Slope", "Grade_Trend_Intercept"]],
                on=["StudentID"],
                how="left",
            )

        count_exsubj = (
            other_lectures_data.groupby(["StudentID", "ClassN"])["Grade"]
            .count()
            .reset_index()
        )
        count_exsubj.rename(columns={"Grade": "Assessment_Count_Others"}, inplace=True)
        finaldf = finaldf.merge(count_exsubj, on=["StudentID", "ClassN"], how="left")

        passed_semester_data = passed_df[(passed_df["ClassN"] == year)].copy()

        grouped_counts = (
            passed_semester_data.groupby(["StudentID", "ClassN"])
            .agg(
                {
                    "Passed": lambda x: (x == 1).sum() if not x.isna().all() else np.nan,
                    "Not_Passed": lambda x: (x == 1).sum() if not x.isna().all() else np.nan,
                }
            )
            .reset_index()
        )
        grouped_counts.columns = ["StudentID", "ClassN", "Total_Passed_Assessments", "Total_Failed_Assessments"]
        finaldf = pd.merge(finaldf, grouped_counts[["StudentID", "ClassN", "Total_Passed_Assessments", "Total_Failed_Assessments"]], on=["StudentID", "ClassN"], how="left")

        target_students = students[
            (students["Subject"] == subject) & (students["ClassN"] == year + pred)
        ]
        target_df = (
            target_students.groupby(["StudentID"])["Grade"]
            .agg(["mean"])
            .reset_index()
        )
        target_df = target_df.rename(columns={"mean": "Target_GPA"})
        finaldf = target_df.merge(finaldf, on=["StudentID"], how="left")

        if "Assessment_Count_Target" in finaldf.columns:
            finaldf = finaldf[finaldf["Assessment_Count_Target"] >= 2]
        finaldf.dropna(subset=["Prior_GPA_Subject"], inplace=True)
        all_years_subject.append(finaldf)

    return pd.concat(all_years_subject, ignore_index=True)

if __name__ == "__main__":
    predictiondataset(years=[5,6,7,8,9,10,11], pred=1, name='nextyear')