import pandas as pd
import os

def concatsemesters(df1, df3):
    df1['Sem1_to_sem2'] = 1
    df3['NextYear_Sem1'] = 1
    return pd.concat([df1, df3], ignore_index=True)

if __name__ == "__main__":
    current_dir = os.getcwd()
    input_dir = os.path.join(current_dir, "data", "prediction_datasets")
    path_sem1 = os.path.join(input_dir, "SEM1TOSEM2.csv")
    path_sem3 = os.path.join(input_dir, "nextyear_1stsemester.csv")
    data1 = pd.read_csv(path_sem1)
    data3 = pd.read_csv(path_sem3)
    combined_data = concatsemesters(data1, data3)
    combined_data['Sem1_to_sem2'] = combined_data['Sem1_to_sem2'].fillna(0).astype(int)
    combined_data['NextYear_Sem1'] = combined_data['NextYear_Sem1'].fillna(0).astype(int)
    output_path = os.path.join(input_dir, "next_semester_combined.csv")
    combined_data.to_csv(output_path, index=False)
    print(f"Combined dataset saved")
