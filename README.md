## Project Structure

```
├── main.py                  # Pipeline orchestrator
├── pipeline_data.py         # Stage 1: Dataset creation
├── pipeline_base.py         # Stage 2: Base model comparison
├── pipeline_catboost.py     # Stage 3: CatBoost training
├── data/
│   ├── students_preprocessed_synthetic.csv
│   └── prediction_datasets/ # Generated datasets
├── results/
│   ├── base_training_results/
│   └── model_files/
└── src/
    ├── functions.py
    ├── ongoing_semester.py
    ├── next_semester.py
    ├── nextyear_firstsemr.py
    ├── nextyear.py
    ├── combine_semesters.py
    ├── basemodels_training.py
    └── catboost_training.py
```

## Prediction Scenarios

| Dataset | Description |
|---|---|
| `{20,30,40,50,60}_ongoing_semester.csv` | Ongoing semester prediction at different day cutoffs |
| `next_semester_combined.csv` | Next semester prediction (Sem1→Sem2 and year-end→next year Sem1) |
| `nextyear.csv` | Next academic year prediction |

## Usage

Run the full pipeline:
```bash
python main.py
```

Run specific steps:
```bash
python main.py --steps 1        # data creation only
python main.py --steps 2        # base model comparison only
python main.py --steps 3        # CatBoost training only
python main.py --steps 1 2      # data creation + base model comparison
```

Each stage can also be run independently:
```bash
python pipeline_data.py
python pipeline_base.py
python pipeline_catboost.py
```