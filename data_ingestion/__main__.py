# ========================
# file: __main__.py  (project root)
# ========================
"""CLI helper: convert raw CSV(s) to canonical Parquet.

Usage examples
--------------
▶ Run button in PyCharm          (expects exactly 1 CSV in ./data_ingestion/samples/)
$ python __main__.py             (same as above)

$ python __main__.py --csv data_ingestion/samples   --symbol AAPL
$ python __main__.py --csv bulk_1998_2024            --symbol AAPL --out data/parquet
"""

from pathlib import Path
import argparse
from data_ingestion.pipelines.csv_pipeline import CSVPipeline
from data_ingestion.persistence import write_parquet

# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #
PROJECT_DIR   = Path(__file__).resolve().parent          # TheFinalProject5/
DEFAULT_CSV   = PROJECT_DIR / "data_ingestion" / "samples"
DEFAULT_OUT   = PROJECT_DIR / "raw" / "minute"

def infer_symbol(csv_path: Path) -> str:
    """AAPL.csv → AAPL,  AAPL_2010.csv → AAPL,  AAPL-minute.csv → AAPL."""
    return csv_path.stem.split("_")[0].split("-")[0].upper()

# ------------------------------------------------------------------ #
# CLI entry-point
# ------------------------------------------------------------------ #
def main() -> None:
    parser = argparse.ArgumentParser("csv-to-parquet")
    parser.add_argument(
        "--csv",
        default=str(DEFAULT_CSV) if DEFAULT_CSV.exists() else None,
        help=f"CSV file or folder (default: {DEFAULT_CSV} if it exists)",
    )
    parser.add_argument(
        "--symbol",
        help="Ticker symbol. If omitted and --csv is a single file, "
             "it will be inferred from the filename prefix."
    )
    parser.add_argument(
        "--out",
        default=str(DEFAULT_OUT),
        help="Output Parquet root (default: raw/minute)",
    )
    args = parser.parse_args()

    if args.csv is None:
        parser.error("--csv is required (no default samples/ folder found)")

    csv_path = Path(args.csv)
    if not csv_path.exists():
        parser.error(f"{csv_path} does not exist")

    # Infer symbol when unambiguous
    if csv_path.is_file() and args.symbol is None:
        args.symbol = infer_symbol(csv_path)

    if args.symbol is None:
        parser.error("--symbol is required when ingesting multiple files")

    # ----------------------------------------------------------------
    pipeline = CSVPipeline()
    files = csv_path.glob("*.csv") if csv_path.is_dir() else [csv_path]

    for f in files:
        df = pipeline.parse(f, symbol=args.symbol)
        write_parquet(df, Path(args.out), partition_cols=["symbol", "year"])
        print(f"Ingested {f.name}  ->  {args.out}")

if __name__ == "__main__":
    main()
