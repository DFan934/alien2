import pandas as pd, tempfile, json, time
from execution.metrics.report import load_blotter, latency_summary

def test_metrics_pipeline():
    # build tiny fake blotter
    rows = []
    for pnl in [10, -5, 20]:
        rows.append(f'{{"ts": {time.time()}, "pnl": {pnl}}},0.5')
    with tempfile.NamedTemporaryFile("w+", delete=False) as fp:
        fp.write("\n".join(rows))
        path = fp.name
    df = load_blotter(path)
    summ = latency_summary(df)
    assert "mean_ms" in summ and summ["mean_ms"] >= 0.0
