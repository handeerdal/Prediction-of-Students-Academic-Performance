# =============================================================================
# AI Assistance Disclosure
# -----------------------------------------------------------------------------
# Tool:    Claude (Sonnet 4.6, Anthropic)
# Date:    2025-04-25
# Prompt:  "Write a CatBoost regression training module with baseline training,
#          Optuna hyperparameter tuning, feature importance-based feature
#          selection (PredictionValuesChange, LossFunctionChange, SHAP), and
#          k-fold cross-validation. Save results, model files, and feature
#          importance as JSON/CSV."
# Notes:   The generated code was reviewed, verified, and partially adjusted
#          by the author. Methodology and evaluation strategy are the
#          author's own design.
# =============================================================================

import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error
from catboost import CatBoostRegressor, Pool
import optuna
from optuna.pruners import MedianPruner

optuna.logging.set_verbosity(optuna.logging.WARNING)

GLOBAL_SEED = 42

def get_tuning_params_by_size(n_samples, n_features):
    if n_samples < 10_000:
        return {
            'iterations':          (200, 400),
            'learning_rate':       (0.03, 0.10),
            'depth':               (3, 5),
            'l2_leaf_reg':         (8.0, 20.0),
            'border_count':        (32, 64),
            'bagging_temperature': (0.3, 1.5),
            'random_strength':     (1.0, 3.0),
            'min_data_in_leaf':    (30, 80),
            'colsample_bylevel':   (0.5, 0.8),
            'early_stopping':      30,
            'n_trials':            20,
        }
    elif n_samples < 100_000:
        return {
            'iterations':          (400, 800),
            'learning_rate':       (0.03, 0.10),
            'depth':               (4, 6),
            'l2_leaf_reg':         (3.0, 10.0),
            'border_count':        (64, 128),
            'bagging_temperature': (1.0, 3.0),
            'random_strength':     (0.5, 2.0),
            'min_data_in_leaf':    (15, 40),
            'colsample_bylevel':   (0.7, 0.95),
            'early_stopping':      50,
            'n_trials':            15,
        }
    else:
        return {
            'iterations':          (600, 1200),
            'learning_rate':       (0.02, 0.08),
            'depth':               (5, 8),
            'l2_leaf_reg':         (1.0, 5.0),
            'border_count':        (128, 255),
            'bagging_temperature': (2.0, 5.0),
            'random_strength':     (0.1, 1.0),
            'min_data_in_leaf':    (5, 20),
            'colsample_bylevel':   (0.8, 1.0),
            'early_stopping':      100,
            'n_trials':            20,
            'timeout':             3600,
        }

def dataprep(df):
    columns_to_drop = ['StudentID', 'TeacherID']
    df = df.drop(columns=[c for c in columns_to_drop if c in df.columns])
    df = df[df['Target_GPA'] > 0]
    df['Total_Passed_Assessments']  = df['Total_Passed_Assessments'].fillna(0)
    df['Total_Failed_Assessments']  = df['Total_Failed_Assessments'].fillna(0)

    df = df.dropna(subset=['Prior_GPA_Subject', 'Target_GPA'])
    df = df[df['Assessment_Count_Target'] >= 2]

    bool_cols = df.select_dtypes(include=['bool']).columns
    if len(bool_cols):
        df[bool_cols] = df[bool_cols].astype(int)

    nan_pct = df.isna().mean() * 100
    df = df.loc[:, nan_pct <= 30].dropna()

    return df


def create_splits(df, target_column):
    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=GLOBAL_SEED)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=GLOBAL_SEED)

    print(f"  Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def mape(y_true, y_pred, eps=0.5):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom  = np.maximum(np.abs(y_true), eps)
    return np.mean(np.abs((y_true - y_pred) / denom)) * 100


def evaluate_model(model, X_tr, y_tr, X_te, y_te, label, feature_list=None):
    if feature_list is not None:
        X_tr = X_tr[feature_list]
        X_te = X_te[feature_list]

    tr_pred = model.predict(X_tr)
    te_pred = model.predict(X_te)

    train_rmse = np.sqrt(np.mean((y_tr - tr_pred) ** 2))
    test_rmse  = np.sqrt(np.mean((y_te - te_pred) ** 2))
    train_r2   = r2_score(y_tr, tr_pred)
    test_r2    = r2_score(y_te, te_pred)
    test_mae   = mean_absolute_error(y_te, te_pred)
    test_mape  = mape(y_te, te_pred)

    print(f"\n{label}:")
    print(f"  Train  RMSE={train_rmse:.4f}  R²={train_r2:.4f}")
    print(f"  Test   RMSE={test_rmse:.4f}  R²={test_r2:.4f}")
    print(f"  Test   MAE={test_mae:.4f}  MAPE={test_mape:.2f}%")
    print(f"  Overfit={test_rmse / train_rmse:.4f}")

    return {
        'train_rmse': train_rmse, 'test_rmse': test_rmse,
        'train_r2':   train_r2,   'test_r2':   test_r2,
        'test_mae':   test_mae,   'test_mape': test_mape,
        'overfit':    test_rmse / train_rmse,
    }


def train_baseline(X_train, y_train, X_val, y_val, cat_cols):
    print("\nBASELINE ")
    n = len(X_train)

    if n < 10_000:
        cfg = dict(iterations=300, learning_rate=0.05, depth=4,
                   l2_leaf_reg=12, min_data_in_leaf=40)
    elif n < 100_000:
        cfg = dict(iterations=600, learning_rate=0.04, depth=5,
                   l2_leaf_reg=6,  min_data_in_leaf=20)
    else:
        cfg = dict(iterations=1000, learning_rate=0.03, depth=6,
                   l2_leaf_reg=3,  min_data_in_leaf=10)

    model = CatBoostRegressor(
        **cfg,
        random_seed=GLOBAL_SEED, verbose=0,
        early_stopping_rounds=50,
        eval_metric='RMSE',
        cat_features=cat_cols,
    )
    model.fit(X_train, y_train, eval_set=(X_val, y_val))
    print(f"  Val RMSE: {model.get_best_score()['validation']['RMSE']:.4f}")
    return model

def tuning(trial, X_train, y_train, X_val, y_val, cat_cols, param_ranges):
    params = {
        'iterations':          trial.suggest_int(  'iterations',          *param_ranges['iterations']),
        'learning_rate':       trial.suggest_float('learning_rate',       *param_ranges['learning_rate'], log=True),
        'depth':               trial.suggest_int(  'depth',               *param_ranges['depth']),
        'l2_leaf_reg':         trial.suggest_float('l2_leaf_reg',         *param_ranges['l2_leaf_reg']),
        'border_count':        trial.suggest_int(  'border_count',        *param_ranges['border_count']),
        'bagging_temperature': trial.suggest_float('bagging_temperature', *param_ranges['bagging_temperature']),
        'random_strength':     trial.suggest_float('random_strength',     *param_ranges['random_strength']),
        'min_data_in_leaf':    trial.suggest_int(  'min_data_in_leaf',    *param_ranges['min_data_in_leaf']),
        'colsample_bylevel':   trial.suggest_float('colsample_bylevel',   *param_ranges['colsample_bylevel']),
        'cat_features':        cat_cols,
        'random_seed':         GLOBAL_SEED,
        'verbose':             0,
        'early_stopping_rounds': param_ranges['early_stopping'],
        'eval_metric':         'RMSE',
    }

    model = CatBoostRegressor(**params)
    model.fit(X_train, y_train, eval_set=(X_val, y_val))

    val_rmse = model.get_best_score()['validation']['RMSE']
    trial.report(val_rmse, step=model.get_best_iteration())
    if trial.should_prune():
        raise optuna.TrialPruned()

    return val_rmse


def train_tuned(X_train, y_train, X_val, y_val, cat_cols):

    print("\n OPTUNA TUNING")
    n, f = len(X_train), X_train.shape[1]
    param_ranges = get_tuning_params_by_size(n, f)
    print(f"  {n:,} sample, {f} feature  →  {param_ranges['n_trials']} trial")

    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=GLOBAL_SEED),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=50, interval_steps=10),
    )

    timeout = param_ranges.get('timeout', None)
    study.optimize(
        lambda t: tuning(t, X_train, y_train, X_val, y_val, cat_cols, param_ranges),
        n_trials=param_ranges['n_trials'],
        timeout=timeout,
        show_progress_bar=True,
    )

    print(f"  Best Val RMSE: {study.best_value:.4f}")

    best_params = study.best_params.copy()

    full_params = best_params.copy()
    full_params.update({
        'cat_features':          cat_cols,
        'random_seed':           GLOBAL_SEED,
        'verbose':               0,
        'early_stopping_rounds': param_ranges['early_stopping'],
        'eval_metric':           'RMSE',
    })

    tuned_model = CatBoostRegressor(**full_params)
    tuned_model.fit(X_train, y_train, eval_set=(X_val, y_val))

    return tuned_model, best_params, study


def select_features(X_train, y_train, X_val, y_val, cat_cols, best_params,
                    early_stopping):
    print("\nFEATURE IMPORTANCE")

    imp_params = best_params.copy()
    imp_params.update({
        'cat_features':          cat_cols,
        'random_seed':           GLOBAL_SEED,
        'verbose':               0,
        'early_stopping_rounds': early_stopping,
        'eval_metric':           'RMSE',
    })
    imp_model = CatBoostRegressor(**imp_params)
    imp_model.fit(X_train, y_train, eval_set=(X_val, y_val))

    train_pool = Pool(X_train, y_train, cat_features=cat_cols)

    pred_changes = imp_model.get_feature_importance(train_pool, type='PredictionValuesChange')
    loss_changes = imp_model.get_feature_importance(train_pool, type='LossFunctionChange')
    shap_values  = imp_model.get_feature_importance(train_pool, type='ShapValues')
    shap_imp     = np.abs(shap_values[:, :-1]).mean(axis=0)

    scaler = MinMaxScaler()
    df_imp = pd.DataFrame({
        'Feature':                X_train.columns,
        'PredictionValuesChange': pred_changes,
        'LossFunctionChange':     loss_changes,
        'SHAP':                   shap_imp,
    })
    for col in ['PredictionValuesChange', 'LossFunctionChange', 'SHAP']:
        df_imp[f'{col}_norm'] = scaler.fit_transform(df_imp[[col]])

    df_imp['Average_Importance'] = df_imp[
        ['PredictionValuesChange_norm', 'LossFunctionChange_norm', 'SHAP_norm']
    ].mean(axis=1)
    df_imp = df_imp.sort_values('Average_Importance', ascending=False).reset_index(drop=True)

    total = len(df_imp)
    counts = sorted(set([total, int(total * 0.8), int(total * 0.7),
                         int(total * 0.6), int(total * 0.5)]), reverse=True)

    print(f"\n  Feature selection ({total} → {counts})")
    results, models = [], {}

    for n_feat in counts:
        top_feats   = df_imp.head(n_feat)['Feature'].tolist()
        cat_filtered = [c for c in cat_cols if c in top_feats]

        sel_params = best_params.copy()
        sel_params.update({
            'cat_features':          cat_filtered,
            'random_seed':           GLOBAL_SEED,
            'verbose':               0,
            'early_stopping_rounds': early_stopping,
            'eval_metric':           'RMSE',
        })

        m = CatBoostRegressor(**sel_params)
        m.fit(X_train[top_feats], y_train,
              eval_set=(X_val[top_feats], y_val))

        val_rmse = m.get_best_score()['validation']['RMSE']
        results.append({'n_features': n_feat, 'val_rmse': val_rmse})
        models[n_feat] = (m, top_feats, cat_filtered)
        print(f"    {n_feat:>3} feature  Val RMSE={val_rmse:.4f}")

    res_df      = pd.DataFrame(results)
    best_n      = int(res_df.loc[res_df['val_rmse'].idxmin(), 'n_features'])
    best_model, best_feats, best_cats = models[best_n]
    print(f"\n  Optimal feature count: {best_n}")

    return best_n, best_feats, best_cats, df_imp, best_model



def cross_validate_winner(df, target_column, winner_params, winner_features,
                          cat_cols_full, dataset_name, output_dir, n_splits=5):

    print(f"\n{n_splits}-FOLD CROSS-VALIDATION")

    X = df[winner_features].copy()
    y = df[target_column].copy()

    cat_filtered = [c for c in cat_cols_full if c in winner_features]

    cv_params = {k: v for k, v in winner_params.items()
                 if k not in ('cat_features', 'verbose', 'random_seed',
                              'early_stopping_rounds', 'eval_metric')}

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=GLOBAL_SEED)

    fold_metrics = []
    for fold_idx, (tr_idx, vl_idx) in enumerate(kf.split(X), 1):
        X_tr, X_vl = X.iloc[tr_idx], X.iloc[vl_idx]
        y_tr, y_vl = y.iloc[tr_idx], y.iloc[vl_idx]

        model = CatBoostRegressor(
            **cv_params,
            cat_features=cat_filtered,
            random_seed=GLOBAL_SEED,
            verbose=0,
        )
        model.fit(X_tr, y_tr)
        preds = model.predict(X_vl)

        rmse      = float(np.sqrt(np.mean((y_vl.values - preds) ** 2)))
        r2        = float(r2_score(y_vl, preds))
        mae       = float(mean_absolute_error(y_vl, preds))
        mape_val  = float(mape(y_vl, preds))

        fold_metrics.append({'fold': fold_idx, 'rmse': rmse, 'r2': r2,
                             'mae': mae, 'mape': mape_val})
        print(f"  Fold {fold_idx}: RMSE={rmse:.4f}  R²={r2:.4f}  "
              f"MAE={mae:.4f}  MAPE={mape_val:.2f}%")

    cv_df = pd.DataFrame(fold_metrics)
    summary = {
        'rmse_mean': float(cv_df['rmse'].mean()), 'rmse_std': float(cv_df['rmse'].std()),
        'r2_mean':   float(cv_df['r2'].mean()),   'r2_std':   float(cv_df['r2'].std()),
        'mae_mean':  float(cv_df['mae'].mean()),  'mae_std':  float(cv_df['mae'].std()),
        'mape_mean': float(cv_df['mape'].mean()), 'mape_std': float(cv_df['mape'].std()),
    }

    print(f"\n  CV Summary (Mean ± Std):")
    print(f"    RMSE = {summary['rmse_mean']:.4f} ± {summary['rmse_std']:.4f}")
    print(f"    R²   = {summary['r2_mean']:.4f} ± {summary['r2_std']:.4f}")
    print(f"    MAE  = {summary['mae_mean']:.4f} ± {summary['mae_std']:.4f}")
    print(f"    MAPE = {summary['mape_mean']:.2f}% ± {summary['mape_std']:.2f}%")

    cv_path = os.path.join(output_dir, f'cv_results_{dataset_name}.json')
    with open(cv_path, 'w') as fp:
        json.dump({
            'dataset':     dataset_name,
            'n_splits':    n_splits,
            'n_features':  len(winner_features),
            'features':    winner_features,
            'fold_results': fold_metrics,
            'summary':     summary,
        }, fp, indent=4)

    return summary


def compare_and_save(
    baseline_model, tuned_model, feat_model,
    X_train, y_train, X_test, y_test,
    all_features, best_features, best_cats,
    output_dir, dataset_name, df_imp,
    split_info, best_params,
):
    print("\nRESULT COMPARISON")

    b_met = evaluate_model(baseline_model, X_train, y_train, X_test, y_test,
                           f"Baseline ({len(all_features)} feat)")
    t_met = evaluate_model(tuned_model,    X_train, y_train, X_test, y_test,
                           f"Tuned ({len(all_features)} feat)")
    f_met = evaluate_model(feat_model,     X_train, y_train, X_test, y_test,
                           f"Final ({len(best_features)} feat)",
                           feature_list=best_features)

    table = pd.DataFrame({
        'Model':     [f'Baseline({len(all_features)})',
                      f'Tuned({len(all_features)})',
                      f'Final({len(best_features)})'],
        'Features':  [len(all_features), len(all_features), len(best_features)],
        'Test_RMSE': [b_met['test_rmse'],  t_met['test_rmse'],  f_met['test_rmse']],
        'Test_R2':   [b_met['test_r2'],    t_met['test_r2'],    f_met['test_r2']],
        'Test_MAE':  [b_met['test_mae'],   t_met['test_mae'],   f_met['test_mae']],
        'Test_MAPE': [b_met['test_mape'],  t_met['test_mape'],  f_met['test_mape']],
        'Overfit':   [b_met['overfit'],    t_met['overfit'],    f_met['overfit']],
    })
    print("\n" + table.to_string(index=False))

    best_idx   = table['Test_MAPE'].idxmin()
    winner_row = table.iloc[best_idx]
    print(f"\nWINNER (MAPE): {winner_row['Model']}")

    if best_idx == 0:
        winner_model, winner_feats = baseline_model, all_features
        winner_params = baseline_model.get_all_params()
    elif best_idx == 1:
        winner_model, winner_feats = tuned_model, all_features
        winner_params = best_params
    else:
        winner_model, winner_feats = feat_model, best_features
        winner_params = best_params

    results_json = {
        'dataset':     dataset_name,
        'split_info':  split_info,
        'winner':      winner_row['Model'],
        'winner_metrics': {
            'test_mape': float(winner_row['Test_MAPE']),
            'test_rmse': float(winner_row['Test_RMSE']),
            'test_r2':   float(winner_row['Test_R2']),
            'test_mae':  float(winner_row['Test_MAE']),
            'overfit':   float(winner_row['Overfit']),
        },
        'best_hyperparameters': {k: (float(v) if isinstance(v, (np.floating,)) else v)
                                  for k, v in best_params.items()},
        'all_models': {
            row['Model']: {k: float(row[k]) for k in
                           ['Features', 'Test_RMSE', 'Test_R2', 'Test_MAE',
                            'Test_MAPE', 'Overfit']}
            for _, row in table.iterrows()
        }
    }
    json_path = os.path.join(output_dir, f'results_{dataset_name}.json')
    with open(json_path, 'w') as fp:
        json.dump(results_json, fp, indent=4, default=str)
    print(f"Results saved: {json_path}")

    model_path = os.path.join(output_dir, f'catboost_final_{dataset_name}.cbm')
    winner_model.save_model(model_path)
    print(f"Model saved: {model_path}  ({len(winner_feats)} feature)")

    feat_path = os.path.join(output_dir, f'features_{dataset_name}.json')
    with open(feat_path, 'w') as fp:
        json.dump({'features': winner_feats, 'cat_features': best_cats}, fp, indent=4)

    imp_path = os.path.join(output_dir, f'feature_importance_{dataset_name}.csv')
    df_imp.to_csv(imp_path, index=False)

    return winner_params, winner_feats


def train_multiple_datasets(datasets, datasets_folder, output_dir,
                            run_cv=True, cv_splits=5):
    overall_summary = []

    for dataset_file in datasets:
        dataset_name = dataset_file.replace('.csv', '')
        print(f"  Dataset: {dataset_name}")

        filepath = os.path.join(datasets_folder, dataset_file)
        data     = pd.read_csv(filepath)
        df       = dataprep(data)

        cat_cols = df.select_dtypes(include=['object']).columns.tolist()
        for col in cat_cols:
            df[col] = df[col].astype(str)

        X_train, X_val, X_test, y_train, y_val, y_test = create_splits(df, 'Target_GPA')
        all_features = X_train.columns.tolist()

        split_info = {
            'n_train': len(X_train), 'n_val': len(X_val), 'n_test': len(X_test),
            'n_features': len(all_features),
            'split_strategy': 'random_record_level_80_16_64',
        }

        baseline_model = train_baseline(X_train, y_train, X_val, y_val, cat_cols)

        tuned_model, best_params, _ = train_tuned(X_train, y_train, X_val, y_val, cat_cols)
        n = len(X_train)
        early_stop = 30 if n < 10_000 else (50 if n < 100_000 else 100)

        best_n, best_feats, best_cats, df_imp, feat_model = select_features(
            X_train, y_train, X_val, y_val, cat_cols, best_params, early_stop
        )

        winner_params, winner_feats = compare_and_save(
            baseline_model, tuned_model, feat_model,
            X_train, y_train, X_test, y_test,
            all_features, best_feats, best_cats,
            output_dir, dataset_name, df_imp,
            split_info, best_params,
        )

        if run_cv:
            cv_summary = cross_validate_winner(
                df=df,
                target_column='Target_GPA',
                winner_params=winner_params,
                winner_features=winner_feats,
                cat_cols_full=cat_cols,
                dataset_name=dataset_name,
                output_dir=output_dir,
                n_splits=cv_splits,
            )
            overall_summary.append({
                'dataset': dataset_name,
                **cv_summary,
            })



if __name__ == "__main__":
    current_dir = os.getcwd()
    input_dir   = os.path.join(current_dir, 'data', 'prediction_datasets')
    output_dir  = os.path.join(current_dir, 'results', 'model_files')
    os.makedirs(output_dir, exist_ok=True)
    datasets = [
        '60_ongoing_semester.csv', '50_ongoing_semester.csv', '40_ongoing_semester.csv',
        '30_ongoing_semester.csv', '20_ongoing_semester.csv',
        'next_semester_combined.csv', 'nextyear.csv'
    ]
    train_multiple_datasets(
        datasets,
        input_dir,
        output_dir,
        run_cv=True,
        cv_splits=5,
    )