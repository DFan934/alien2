from verify_loader import run_phase1_checks, pretty_print_report

report = run_phase1_checks(
    root="../parquet",
    symbols=["RRC", "BBY"],
    start="1998-08-01 09:30",
    end="1999-02-01 16:00",
    columns=None,  # or a minimal projection if you want
    speed_target_s_per_million=2.0,
)

pretty_print_report(report)
