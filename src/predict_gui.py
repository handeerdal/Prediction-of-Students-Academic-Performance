# =============================================================================
# AI Assistance Disclosure
# -----------------------------------------------------------------------------
# Tool:    Claude (Opus 4.8, Anthropic)
# Date:    2026-06-07
# Prompt:  "Desktop window to predict GPA. Inputs: scenario (20/30/.. days,
#          next semester, next year), semester/type, subject, student ID(s).
#          Show class number, class section and class group as extra COLUMNS in
#          the results (not as inputs); show '-' when a dataset lacks them
#          (e.g. next-year has no section/group)."
# Notes:   Demo front-end over the author's trained models and the prediction
#          datasets produced by the pipeline. Models and feature engineering
#          are the author's own work. Reviewed and adjusted by the author.
# =============================================================================
#
# Run from the PROJECT ROOT (same place as main.py), after:
#   python main.py --steps 1 3
#
#   python predict_gui.py

import os
import json
import tkinter as tk
from tkinter import ttk
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor

DATASET_DIR = os.path.join("data", "prediction_datasets")
MODEL_DIR   = os.path.join("results", "model_files")
RISK_THRESHOLD = 6.0

SCENARIOS = {
    "Ongoing semester - 20 days": "20_ongoing_semester",
    "Ongoing semester - 30 days": "30_ongoing_semester",
    "Ongoing semester - 40 days": "40_ongoing_semester",
    "Ongoing semester - 50 days": "50_ongoing_semester",
    "Ongoing semester - 60 days": "60_ongoing_semester",
    "Next semester":              "next_semester_combined",
    "Next year":                  "nextyear",
}
SUBJECTS = ["Lithuanian Language", "Mathematics", "History", "Geography",
            "Biology", "English Language", "Physics", "Chemistry"]

SEM_ONGOING = ["All semesters", "Semester 1", "Semester 2"]
SEM_NEXTSEM = ["Sem 1 -> Sem 2", "Year-end -> Next year Sem 1"]
SEM_NONE    = ["All"]

BG = "#f4f6f8"; CARD = "#ffffff"; ACCENT = "#2E86AB"; TEXT = "#1f2933"
OK_CLR = "#1f8a4c"; RISK_CLR = "#c0392b"; MUTED = "#6b7280"

_cache = {}


def _norm(series):
    return series.astype(str).str.replace(r"\.0$", "", regex=True)


def _cell(row, col):
    if col in row and pd.notna(row[col]):
        return str(row[col]).replace(".0", "")
    return "-"


def load_scenario(scenario):
    if scenario in _cache:
        return _cache[scenario]
    csv_path   = os.path.join(DATASET_DIR, f"{scenario}.csv")
    model_path = os.path.join(MODEL_DIR, f"catboost_final_{scenario}.cbm")
    feat_path  = os.path.join(MODEL_DIR, f"features_{scenario}.json")
    for p in (csv_path, model_path, feat_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing: {p}  (run: python main.py --steps 1 3)")
    df = pd.read_csv(csv_path)
    df["__sid"] = _norm(df["StudentID"])
    model = CatBoostRegressor(); model.load_model(model_path)
    with open(feat_path) as fp:
        meta = json.load(fp)
    bundle = (df, model, meta["features"], meta.get("cat_features", []))
    _cache[scenario] = bundle
    return bundle


def predict_rows(df, model, features, cat_features):
    X = df[features].copy()
    for c in cat_features:
        if c in X.columns:
            X[c] = X[c].astype(str)
    num = [c for c in X.columns if c not in cat_features]
    X[num] = X[num].apply(pd.to_numeric, errors="coerce").fillna(0)
    return model.predict(X)


def split_ids(text):
    return [t.strip().replace(".0", "") for t in text.split(",") if t.strip()]


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Student Performance Predictor")
        self.geometry("980x680")
        self.configure(bg=BG)

        st = ttk.Style(self); st.theme_use("clam")
        st.configure("TLabel", background=BG, foreground=TEXT, font=("Segoe UI", 10))
        st.configure("Card.TLabel", background=CARD, foreground=TEXT, font=("Segoe UI", 10))
        st.configure("Header.TLabel", background=BG, foreground=ACCENT, font=("Segoe UI Semibold", 18))
        st.configure("Sub.TLabel", background=BG, foreground=MUTED, font=("Segoe UI", 10))
        st.configure("Accent.TButton", font=("Segoe UI Semibold", 10), padding=6,
                     foreground="white", background=ACCENT)
        st.map("Accent.TButton", background=[("active", "#246b8a")])
        st.configure("Treeview", rowheight=26, font=("Segoe UI", 10),
                     fieldbackground=CARD, background=CARD)
        st.configure("Treeview.Heading", font=("Segoe UI Semibold", 10))

        ttk.Label(self, text="Student Performance Predictor", style="Header.TLabel").pack(
            anchor="w", padx=20, pady=(18, 0))
        ttk.Label(self, text="Pick scenario, semester/type, subject and student ID(s).",
                  style="Sub.TLabel").pack(anchor="w", padx=20, pady=(0, 12))

        f = tk.Frame(self, bg=CARD, highlightthickness=1, highlightbackground="#e1e5ea")
        f.pack(fill="x", padx=20)
        f.columnconfigure(0, weight=1); f.columnconfigure(1, weight=1)

        def lab(text, r, c):
            ttk.Label(f, text=text, style="Card.TLabel").grid(
                row=r, column=c, sticky="w", padx=12, pady=(10, 2))

        lab("Prediction", 0, 0)
        self.scenario_cb = ttk.Combobox(f, values=list(SCENARIOS.keys()), state="readonly")
        self.scenario_cb.current(1)
        self.scenario_cb.grid(row=1, column=0, sticky="ew", padx=12, pady=(0, 8))
        self.scenario_cb.bind("<<ComboboxSelected>>", self.on_scenario_change)

        lab("Subject", 0, 1)
        self.subject_cb = ttk.Combobox(f, values=SUBJECTS, state="readonly")
        self.subject_cb.current(1)
        self.subject_cb.grid(row=1, column=1, sticky="ew", padx=12, pady=(0, 8))

        lab("Semester / Type", 2, 0)
        self.semester_cb = ttk.Combobox(f, values=SEM_ONGOING, state="readonly")
        self.semester_cb.current(0)
        self.semester_cb.grid(row=3, column=0, sticky="ew", padx=12, pady=(0, 8))

        lab("Student ID(s) - comma-separated, blank = first 15", 4, 0)
        self.id_entry = ttk.Entry(f, font=("Segoe UI", 10))
        self.id_entry.grid(row=5, column=0, columnspan=2, sticky="ew", padx=12, pady=(0, 10))

        ttk.Button(f, text="Predict", style="Accent.TButton", command=self.on_predict).grid(
            row=6, column=0, columnspan=2, sticky="e", padx=12, pady=(0, 12))

        cols = ("student", "classn", "section", "group", "subject", "scope",
                "predicted", "actual", "error", "status")
        heads = ("Student", "ClassN", "Section", "Group", "Subject", "Sem/Type",
                 "Predicted", "Actual", "Error", "Status")
        widths = (65, 55, 60, 65, 100, 90, 80, 65, 50, 175)
        wrap = tk.Frame(self, bg=BG); wrap.pack(fill="both", expand=True, padx=20, pady=14)
        self.tree = ttk.Treeview(wrap, columns=cols, show="headings")
        for c, h, w in zip(cols, heads, widths):
            self.tree.heading(c, text=h)
            self.tree.column(c, width=w, anchor=("w" if c in ("subject", "status") else "center"))
        self.tree.tag_configure("risk", foreground=RISK_CLR)
        self.tree.tag_configure("ok", foreground=OK_CLR)
        vsb = ttk.Scrollbar(wrap, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        self.tree.pack(side="left", fill="both", expand=True); vsb.pack(side="right", fill="y")

        self.status = ttk.Label(self, text="Ready.", style="Sub.TLabel")
        self.status.pack(anchor="w", padx=20, pady=(0, 12))

        self.on_scenario_change()

    def on_scenario_change(self, event=None):
        scen = SCENARIOS[self.scenario_cb.get()]
        if scen.endswith("ongoing_semester"):
            opts = SEM_ONGOING
        elif scen == "next_semester_combined":
            opts = SEM_NEXTSEM
        else:
            opts = SEM_NONE
        self.semester_cb.config(values=opts); self.semester_cb.current(0)

    def on_predict(self):
        self.tree.delete(*self.tree.get_children())
        label = self.scenario_cb.get(); subject = self.subject_cb.get()
        scope = self.semester_cb.get(); scenario = SCENARIOS[label]

        try:
            df, model, features, cat_features = load_scenario(scenario)
        except FileNotFoundError as e:
            self.status.config(text=str(e), foreground=RISK_CLR); return

        sub = df[df["Subject"] == subject].copy() if "Subject" in df.columns else df.copy()

        if scenario.endswith("ongoing_semester") and "Semester" in sub.columns:
            semnum = sub["Semester"].astype(str).str.split("/").str[0]
            if scope == "Semester 1":
                sub = sub[semnum == "1"]
            elif scope == "Semester 2":
                sub = sub[semnum == "2"]
        elif scenario == "next_semester_combined":
            if scope.startswith("Sem 1") and "Sem1_to_sem2" in sub.columns:
                sub = sub[sub["Sem1_to_sem2"] == 1]
            elif scope.startswith("Year-end") and "NextYear_Sem1" in sub.columns:
                sub = sub[sub["NextYear_Sem1"] == 1]

        ids = split_ids(self.id_entry.get())
        sub = sub[sub["__sid"].isin(ids)] if ids else sub.head(15)

        if sub.empty:
            self.status.config(text=f"No matching records in {label} for those filters.",
                               foreground=RISK_CLR); return

        preds = predict_rows(sub, model, features, cat_features)
        has_actual = "Target_GPA" in sub.columns
        errs = []
        for i, (_, row) in enumerate(sub.iterrows()):
            p = float(preds[i])
            a = float(row["Target_GPA"]) if has_actual else float("nan")
            err = abs(p - a) if (has_actual and not np.isnan(a)) else None
            if err is not None:
                errs.append(err)
            tag = "risk" if p < RISK_THRESHOLD else "ok"
            status = "AT RISK - intervene early" if p < RISK_THRESHOLD else "On track"
            scope_cell = (str(row["Semester"]) if scenario.endswith("ongoing_semester")
                          and "Semester" in sub.columns else scope)
            self.tree.insert("", "end",
                values=(row["__sid"], _cell(row, "ClassN"), _cell(row, "ClassSection"),
                        _cell(row, "ClassGroup"), subject, scope_cell, f"{p:.2f}",
                        f"{a:.2f}" if (has_actual and not np.isnan(a)) else "-",
                        f"{err:.2f}" if err is not None else "-", status),
                tags=(tag,))

        msg = f"{len(sub)} prediction(s) - {label} - {scope} - {subject}"
        if errs:
            msg += f"  |  mean error: {np.mean(errs):.2f} GPA points"
        self.status.config(text=msg, foreground=MUTED)


if __name__ == "__main__":
    App().mainloop()