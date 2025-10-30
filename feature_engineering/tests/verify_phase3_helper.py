from pathlib import Path
from feature_engineering.tests.verify_phase1_part3 import run_phase3_checks, pretty_print

# this file lives in feature_engineering/tests/, so:
HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]                     # .../TheFinalProject5/
INPUT_ROOT = ROOT / "parquet"              # .../TheFinalProject5/parquet
OUTPUT_ROOT = ROOT / "feature_engineering" / "feature_parquet"

r = run_phase3_checks(
    INPUT_ROOT,
    OUTPUT_ROOT,
    symbols=["RRC","BBY"],
    start="1998-08-01",
    end="1999-02-01",
    normalization="per_symbol",
)
pretty_print(r)
