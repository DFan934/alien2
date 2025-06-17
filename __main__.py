# file: __main__.py  (project root: TheFinalProject5/__main__.py)
from pathlib import Path
import argparse
from data_ingestion.pipelines.csv_pipeline import CSVPipeline
from data_ingestion.persistence import write_parquet

PROJECT_DIR = Path(__file__).resolve().parent           # TheFinalProject5/
DEFAULT_CSV = PROJECT_DIR / "data_ingestion" / "samples"
DEFAULT_OUT = PROJECT_DIR / "raw" / "minute"


def infer_symbol(csv_path: Path) -> str:
    """AAPL.csv → AAPL ,  AAPL_2010.csv → AAPL."""
    return csv_path.stem.split("_")[0].split("-")[0].upper()


def main() -> None:
    p = argparse.ArgumentParser("csv-to-parquet")
    p.add_argument("--csv",
                   default=str(DEFAULT_CSV) if DEFAULT_CSV.exists() else None,
                   help=f"CSV file or folder (default: {DEFAULT_CSV} if it exists)")
    p.add_argument("--symbol",
                   help="Ticker. If omitted and exactly one CSV is provided, it "
                        "is inferred from the filename.")
    p.add_argument("--out", default=str(DEFAULT_OUT),
                   help="Output Parquet root (default: raw/minute)")
    args = p.parse_args()

    if args.csv is None:
        p.error("--csv is required (no default samples/ folder found)")

    csv_path = Path(args.csv)
    if not csv_path.exists():
        p.error(f"{csv_path} does not exist")

    # ------------------------------------------------------------------
    # Build the list of files **before** deciding if we can infer symbol
    # ------------------------------------------------------------------
    files = list(csv_path.glob("*.csv")) if csv_path.is_dir() else [csv_path]

    if args.symbol is None and len(files) == 1:
        args.symbol = infer_symbol(files[0])

    if args.symbol is None:
        p.error("--symbol is required when ingesting multiple files")

    pipeline = CSVPipeline()
    out_root = Path(args.out)

    for f in files:
        df = pipeline.parse(f, symbol=args.symbol)
        write_parquet(df, out_root, partition_cols=["symbol", "year"])
        print(f"Ingested {f.name}  →  {out_root}")


if __name__ == "__main__":
    main()
