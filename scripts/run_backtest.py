"""
Regenerates every evaluation metric in this project: runs the chronological
holdout backtest (Brier score, log loss, calibration, ECE) for Dixon-Coles,
the Elo+Attack/Defense XGBoost model, and three baselines, then writes a
JSON report and a calibration plot to reports/.

Usage (from the project root):
    python3 scripts/run_backtest.py
"""
import json
import random
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from backend.competitions.registry import get_competition  # noqa: E402
from backend.config import CONFIG  # noqa: E402
from backend.evaluation.backtest import run_backtest  # noqa: E402

REPORTS_DIR = REPO_ROOT / "reports"


def main():
    seed = CONFIG["evaluation"]["random_seed"]
    random.seed(seed)
    np.random.seed(seed)

    competition = get_competition("world_cup")
    raw_df = competition.data_source.load()

    report = run_backtest(
        raw_df,
        ratings_cfg=CONFIG["ratings"],
        model_cfg=CONFIG["model"],
        dixon_coles_cfg=CONFIG["dixon_coles"],
        elo_k_overrides=competition.elo_k_overrides,
        holdout_start_date=CONFIG["evaluation"]["holdout_start_date"],
        calibration_bins=CONFIG["evaluation"]["calibration_bins"],
    )

    REPORTS_DIR.mkdir(exist_ok=True)
    metrics_path = REPORTS_DIR / "backtest_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Holdout: {report['holdout_start_date']} onward, refit every period ({report['refit_period']})")
    print(f"Periods evaluated: {len(report['periods_evaluated'])}\n")

    rows = sorted(report["models"].items(), key=lambda kv: kv[1]["brier_score"])
    header = f"{'model':<16} {'n':>6} {'brier':>8} {'log_loss':>9} {'ECE':>7}"
    print(header)
    print("-" * len(header))
    for name, m in rows:
        print(f"{name:<16} {m['n_matches']:>6} {m['brier_score']:>8.4f} {m['log_loss']:>9.4f} {m['expected_calibration_error']:>7.4f}")

    print(f"\nWrote {metrics_path}")
    _plot_calibration(report, REPORTS_DIR / "calibration_curve.png")


def _plot_calibration(report, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfectly calibrated")

    for name, m in report["models"].items():
        bins = [b for b in m["calibration_curve"] if b["count"] > 0]
        if not bins:
            continue
        x = [b["mean_predicted"] for b in bins]
        y = [b["empirical_freq"] for b in bins]
        ax.plot(x, y, marker="o", label=name)

    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Empirical frequency")
    ax.set_title("Reliability diagram (pooled one-vs-rest)")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
