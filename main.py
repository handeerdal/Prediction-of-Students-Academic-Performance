# =============================================================================
# AI Assistance Disclosure
# -----------------------------------------------------------------------------
# Tool:    Claude (Sonnet 4.6, Anthropic)
# Date:    2025-05-12
# Prompt:  "Write a simple main.py that runs my three pipeline scripts
#          in order, with the option to call different steps."
# Notes:   The underlying pipeline scripts were written by the author. AI was
#          used only to write this small orchestrator that runs them in
#          sequence with a step-selection argument. The generated code was
#          reviewed and partially adjusted by the author.
# =============================================================================

# Usage:
#   python main.py              # run all steps
#   python main.py --steps 1    # only data creation
#   python main.py --steps 2    # only base model training
#   python main.py --steps 3    # only catboost training
#   python main.py --steps 1 2  # data creation + base training
 
import argparse
import runpy
 
parser = argparse.ArgumentParser()
parser.add_argument("--steps", nargs="+", type=int, choices=[1, 2, 3], default=[1, 2, 3])
args = parser.parse_args()
 
if 1 in args.steps:
    print("\n=== STEP 1: Data Creation ===")
    runpy.run_path("pipeline_data.py")
 
if 2 in args.steps:
    print("\n=== STEP 2: Base Model Training ===")
    runpy.run_path("pipeline_basemodeltrain.py")
 
if 3 in args.steps:
    print("\n=== STEP 3: CatBoost Training ===")
    runpy.run_path("pipeline_catboost.py")
 
print("\n=== DONE ===")