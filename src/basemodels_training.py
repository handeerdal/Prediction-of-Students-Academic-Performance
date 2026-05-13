import os
import json
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor,
    ExtraTreesRegressor, BaggingRegressor
)
from sklearn.neighbors import KNeighborsRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

warnings.filterwarnings("ignore")


def dataprep(df):
    columns_to_drop = ['StudentID', 'ClassGroup', 'TeacherID', 'Semester']
    df['SchoolID'] = df['SchoolID'].astype(str)
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    df = df.fillna({'Total_Passed_Assessments': 0, 'Total_Failed_Assessments': 0})
    df = df.dropna(subset=['Prior_GPA_Subject', 'Target_GPA'])
    df = df[df['Assessment_Count_Target'] >= 2]

    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    bool_cols = df.select_dtypes(include=['bool']).columns
    if len(bool_cols) > 0:
        df[bool_cols] = df[bool_cols].astype(int)

    nan_percent = df.isna().mean() * 100
    df = df.loc[:, nan_percent <= 30]
    df = df.dropna()
    return df


def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred  = model.predict(X_test)

    return {
        'train_r2':   r2_score(y_train, y_train_pred),
        'test_r2':    r2_score(y_test,  y_test_pred),
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'test_rmse':  np.sqrt(mean_squared_error(y_test,  y_test_pred)),
        'train_mae':  mean_absolute_error(y_train, y_train_pred),
        'test_mae':   mean_absolute_error(y_test,  y_test_pred),
    }


def compare_models(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    models = {
        'Linear Regression': LinearRegression(),
        'Extra Trees':       ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'AdaBoost':          AdaBoostRegressor(n_estimators=100, random_state=42),
        'Random Forest':     RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'LightGBM':          LGBMRegressor(n_estimators=100, random_state=42, n_jobs=-1, verbose=-1),
        'CatBoost':          CatBoostRegressor(n_estimators=100, random_state=42, verbose=0),
        'KNN':               KNeighborsRegressor(),
        'Bagging':           BaggingRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    }

    results = {}
    n = len(models)
    for i, (name, model) in enumerate(models.items(), start=1):
        print(f"  [{i}/{n}] {name:<20s} training...", end='\r', flush=True)
        t0 = time.time()
        results[name] = evaluate_model(model, X_train, X_test, y_train, y_test)
        elapsed = time.time() - t0
        print(f"  [{i}/{n}] {name:<20s} done  "
              f"R²={results[name]['test_r2']:.4f}  "
              f"RMSE={results[name]['test_rmse']:.4f}  "
              f"({elapsed:.1f}s)")

    results_df = pd.DataFrame(results).T
    results_df = results_df.sort_values('test_r2', ascending=False)
    return results_df


def train_multiple_datasets(dataset_names, datasets_folder, output_dir):
    all_results = {}

    for dataset_file in dataset_names:
        dataset_name = dataset_file.replace('.csv', '')
        print(f"  Dataset: {dataset_name}")

        filepath = os.path.join(datasets_folder, dataset_file)
        data = pd.read_csv(filepath)
        df = dataprep(data)

        X = df.drop('Target_GPA', axis=1)
        y = df['Target_GPA']

        results_df = compare_models(X, y, test_size=0.2, random_state=42)

        best_model_name = results_df['test_r2'].idxmax()
        print(f"\n  Best: {best_model_name} "
              f"(R²={results_df.loc[best_model_name, 'test_r2']:.4f}, "
              f"RMSE={results_df.loc[best_model_name, 'test_rmse']:.4f})")

        all_results[dataset_name] = {
            'all_models': results_df.to_dict(orient='index'),
            'best_model': {
                'name':    best_model_name,
                'metrics': results_df.loc[best_model_name].to_dict(),
            },
        }

    filepath = os.path.join(output_dir, "all_base_results.json")
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)
    print(f"\nResults saved: {filepath}")

if __name__ == "__main__":
    current_dir = os.getcwd()
    input_dir   = os.path.join(current_dir, "data", "prediction_datasets")
    output_dir  = os.path.join(current_dir, "results", "base_training_results")
    os.makedirs(output_dir, exist_ok=True)
    datasets = [
    '20_ongoing_semester.csv',
    '30_ongoing_semester.csv',
    '40_ongoing_semester.csv',
    '50_ongoing_semester.csv',
    '60_ongoing_semester.csv',
    'next_semester_combined.csv',
    'nextyear.csv',]
    train_multiple_datasets(datasets, input_dir, output_dir)